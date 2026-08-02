# MODULUS-ELF Colab TPU Runner

This runner is for the first practical test loop on a Colab-accessible TPU. It is intentionally staged so that the official ELF OpenWebText baseline is checked before any MODULUS geometry result is admitted.

## Notebook

Use:

```text
notebooks/MODULUS_ELF_Colab_TPU_Runner.ipynb
```

Open it in Google Colab, then select:

```text
Runtime -> Change runtime type -> TPU
```

Run the notebook in order.

## Test sequence

1. Probe the TPU runtime and print `jax.devices()`.
2. Clone upstream `lillian039/ELF` and this overlay repository.
3. Install upstream ELF requirements.
4. Run a smoke-sized official ELF-B OpenWebText checkpoint generation pass.
5. Optionally run the paper-comparable ELF-B evaluation with 1,000 samples and online PPL.
6. Apply overlay files into the upstream checkout.
7. Run the MODULUS geometry helper smoke test on TPU.

## Admission gates

The first admitted result must be the official upstream baseline. No MODULUS comparison is meaningful until the released checkpoint reproduces within expected tolerance.

Baseline command:

```bash
cd /content/elf-upstream/src
python eval.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-owt \
  --config_override use_wandb=false
```

For Colab initialization, the notebook first uses a small smoke profile:

```text
num_samples=8
online_eval=false
global_batch_size=8
sampling_configs_path=configs/sampling_configs/colab_smoke.yml
```

The paper-comparable profile uses the upstream sampling config and `num_samples=1000` with online PPL enabled.

## Expected baseline reference

The upstream README reports ELF-B OpenWebText checkpoint evaluation around Gen. PPL 24.1 and entropy 5.15 with 32-step SDE sampling. The notebook should not treat smoke outputs as comparable to that number; smoke only tests runtime plumbing.

## Known limitations

- Colab TPU memory and package state can change. Restart the runtime after dependency installation if JAX cannot see TPU devices.
- Full paper-comparable evaluation may exceed a short interactive Colab session.
- The current overlay smoke only validates geometry helper execution on TPU. Full `hybrid_sphere` training requires wiring `modulus_geometry.py` into upstream `sampling_utils.py`, `train_step.py`, and `configs/config.py`.
