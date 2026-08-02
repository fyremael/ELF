# MODULUS-ELF Patch Points

This document records the minimal patch surface identified from the official ELF JAX source.

## Primary training seam

File: `src/train_step.py`

The upstream training step:

1. encodes tokens into frozen T5 latent space as `x0`;
2. samples Gaussian noise with the same shape as `x0`;
3. constructs `denoiser_z` with `add_noise(x0, noise, t, config, cond_seq_mask=...)`;
4. trains an x-prediction network output converted to velocity with `net_out_to_v_x`;
5. uses a separate decoder branch at `t=1` for token CE loss.

The MODULUS patch should add a latent-geometry adapter at this seam, not inside the Transformer blocks first.

## Primary sampling seam

File: `src/utils/sampling_utils.py`

The upstream flow utilities define:

- `add_noise(x0, noise, t, config, cond_seq_mask=None)`
- `net_out_to_v_x(net_out, z, t, t_eps=...)`
- `_ode_step(...)`
- `_sde_step(...)`

The first overlay should add geometry-aware versions while preserving upstream behavior when disabled:

```python
if getattr(config, "latent_geometry", "off") == "off":
    return upstream_behavior(...)
```

## Generation seam

File: `src/utils/generation_utils.py`

The upstream generator:

1. constructs initial latent noise in `_shard_noise`;
2. scans `_ode_step` or `_sde_step` over sampled timesteps;
3. performs a final ODE step;
4. decodes latents through `_dlm_decode_batch`.

MODULUS sampling must ensure that generated latents remain on the chosen geometry before final decoding.

## Config seam

File: `src/configs/config.py`

Add these fields with behavior-preserving defaults:

```python
latent_geometry: str = "off"          # off | sphere | hybrid_sphere | channel_sphere
modulus_eps: float = 1e-6
modulus_preserve_radius: bool = True
modulus_project_velocity: bool = True
modulus_log_diagnostics: bool = True
```

## First admissible branch

Use `hybrid_sphere` first:

```text
x = r * u
r = stop_gradient(norm(x)) or batch/token radius statistic
u = x / max(norm(x), eps)
```

Then train angularly controlled flow while preserving radius.

## Deferred interventions

Do not include these in WP00 unless the baseline and hybrid path are already stable:

- replacing model internals with nGPT-style spherical blocks;
- changing RoPE;
- changing tokenizer or T5 encoder;
- changing Muon optimizer internals;
- using audio latents;
- adding new evaluators.

Those belong to later work packages.
