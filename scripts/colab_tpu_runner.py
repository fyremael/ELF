#!/usr/bin/env python3
"""Colab TPU runner for initial MODULUS-ELF testing.

Run from a Colab TPU runtime after cloning this repository, for example:

    python /content/modulus-elf-overlay/scripts/colab_tpu_runner.py prepare
    python /content/modulus-elf-overlay/scripts/colab_tpu_runner.py probe
    python /content/modulus-elf-overlay/scripts/colab_tpu_runner.py baseline-smoke
    python /content/modulus-elf-overlay/scripts/colab_tpu_runner.py geometry-smoke

The smoke path is not a paper-comparable OpenWebText evaluation. It verifies
runtime plumbing before a full ELF-B checkpoint evaluation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content")
UPSTREAM = ROOT / "elf-upstream"
OVERLAY = ROOT / "modulus-elf-overlay"
SRC = UPSTREAM / "src"

UPSTREAM_URL = "https://github.com/lillian039/ELF.git"
OVERLAY_URL = "https://github.com/fyremael/ELF.git"
OVERLAY_BRANCH = "agent/modulus-elf-jax-overlay"


def run(cmd: list[str | Path], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(map(str, cmd))
    print(f"+ {printable}", flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.check_call([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=merged_env)


def clone_or_refresh(path: Path, url: str, branch: str | None = None, refresh: bool = False) -> None:
    if refresh and path.exists():
        shutil.rmtree(path)
    if path.exists():
        print(f"Using existing checkout: {path}")
        return
    cmd: list[str | Path] = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([url, path])
    run(cmd)


def prepare(refresh: bool = False, install: bool = True) -> None:
    clone_or_refresh(UPSTREAM, UPSTREAM_URL, refresh=refresh)
    clone_or_refresh(OVERLAY, OVERLAY_URL, branch=OVERLAY_BRANCH, refresh=refresh)
    if install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=UPSTREAM)
    print("Prepared:")
    print(f"  upstream={UPSTREAM}")
    print(f"  overlay={OVERLAY}")


def probe() -> None:
    print("Python:", sys.version)
    print("COLAB_TPU_ADDR:", os.environ.get("COLAB_TPU_ADDR"))
    print("TPU_NAME:", os.environ.get("TPU_NAME"))
    import jax

    print("JAX:", jax.__version__)
    print("JAX backend:", jax.default_backend())
    devices = jax.devices()
    print("JAX devices:", devices)
    if not any(getattr(d, "platform", None) == "tpu" for d in devices):
        raise SystemExit(
            "No TPU devices visible to JAX. In Colab select Runtime -> Change runtime type -> TPU, "
            "then restart and rerun."
        )


def write_smoke_sampling_config() -> Path:
    cfg = SRC / "configs" / "sampling_configs" / "colab_smoke.yml"
    cfg.write_text(
        "- sampling_method: sde\n"
        "  num_sampling_steps: [4]\n"
        "  cfgs: [1]\n"
        "  sde_gamma: 0.5\n"
        "  self_cond_cfg_scales: [1]\n"
        "  time_schedule: uniform\n",
        encoding="utf-8",
    )
    print(cfg.read_text(encoding="utf-8"))
    return cfg


def baseline_smoke() -> None:
    write_smoke_sampling_config()
    env = {"WANDB_MODE": "disabled"}
    run(
        [
            sys.executable,
            "eval.py",
            "--config",
            "configs/training_configs/train_owt_ELF-B.yml",
            "--checkpoint_path",
            "embedded-language-flows/ELF-B-owt",
            "--config_override",
            "use_wandb=false",
            "--config_override",
            "online_eval=false",
            "--config_override",
            "num_samples=8",
            "--config_override",
            "global_batch_size=8",
            "--config_override",
            "sampling_configs_path=configs/sampling_configs/colab_smoke.yml",
            "--config_override",
            "output_dir=/content/elf-colab-smoke",
        ],
        cwd=SRC,
        env=env,
    )


def full_eval() -> None:
    env = {"WANDB_MODE": "disabled"}
    run(
        [
            sys.executable,
            "eval.py",
            "--config",
            "configs/training_configs/train_owt_ELF-B.yml",
            "--checkpoint_path",
            "embedded-language-flows/ELF-B-owt",
            "--config_override",
            "use_wandb=false",
            "--config_override",
            "num_samples=1000",
            "--config_override",
            "output_dir=/content/elf-b-owt-full-eval",
        ],
        cwd=SRC,
        env=env,
    )


def apply_overlay() -> None:
    run(["bash", OVERLAY / "scripts" / "apply_overlay_to_upstream.sh", UPSTREAM])


def geometry_smoke() -> None:
    apply_overlay()
    code = r'''
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from utils.modulus_geometry import (
    decompose_radius_direction,
    project_to_tangent,
    hybrid_sphere_interpolate,
)

key = jax.random.PRNGKey(0)
x0 = jax.random.normal(key, (2, 16, 512))
noise = jax.random.normal(jax.random.fold_in(key, 1), x0.shape)
t = jnp.array([0.25, 0.75], dtype=jnp.float32)

@jax.jit
def smoke(x0, noise, t):
    direction, radius = decompose_radius_direction(x0)
    tangent_noise = project_to_tangent(noise, direction)
    z = hybrid_sphere_interpolate(x0, tangent_noise, t, noise_scale=2.0)
    z_dir, z_radius = decompose_radius_direction(z)
    return {
        "direction_norm_mean": jnp.mean(jnp.linalg.norm(z_dir, axis=-1)),
        "radius_mean": jnp.mean(z_radius),
        "z_finite": jnp.all(jnp.isfinite(z)),
    }

out = smoke(x0, noise, t)
print(jax.tree_util.tree_map(lambda x: x.item() if x.shape == () else x, out))
assert bool(out["z_finite"]), "Non-finite values in geometry smoke test."
print("MODULUS geometry helper smoke test passed on:", jax.default_backend())
'''
    run([sys.executable, "-c", code], cwd=SRC)


def main() -> None:
    parser = argparse.ArgumentParser(description="MODULUS-ELF Colab TPU runner")
    parser.add_argument(
        "stage",
        choices=["prepare", "probe", "baseline-smoke", "geometry-smoke", "full-eval", "all-smoke"],
    )
    parser.add_argument("--refresh", action="store_true", help="Delete and reclone checkouts during prepare")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install during prepare")
    args = parser.parse_args()

    if args.stage == "prepare":
        prepare(refresh=args.refresh, install=not args.no_install)
    elif args.stage == "probe":
        probe()
    elif args.stage == "baseline-smoke":
        baseline_smoke()
    elif args.stage == "geometry-smoke":
        geometry_smoke()
    elif args.stage == "full-eval":
        full_eval()
    elif args.stage == "all-smoke":
        prepare(refresh=args.refresh, install=not args.no_install)
        probe()
        baseline_smoke()
        geometry_smoke()


if __name__ == "__main__":
    main()
