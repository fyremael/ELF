# Codex prompt: MODULUS-ELF JAX overlay

You are working in the official ELF JAX codebase cloned from `lillian039/ELF`. The target overlay repository is `fyremael/ELF`.

## Mission

Implement the first MODULUS-ELF overlay while preserving official ELF behavior when `latent_geometry=off`.

## Required sequence

1. Reproduce the official ELF-B OpenWebText checkpoint evaluation using the upstream README command.
2. Record the upstream commit SHA, command, hardware/backend, seed, Gen. PPL, entropy, and output path.
3. Add config fields to `src/configs/config.py`:

```python
latent_geometry: str = "off"
modulus_eps: float = 1e-6
modulus_preserve_radius: bool = True
modulus_project_velocity: bool = True
modulus_log_diagnostics: bool = True
```

4. Add `src/utils/modulus_geometry.py` from this overlay.
5. Modify `src/utils/sampling_utils.py` minimally:
   - import the geometry helpers;
   - keep upstream `add_noise` behavior exactly when `latent_geometry == "off"`;
   - route `sphere`, `hybrid_sphere`, and `channel_sphere` through the helper;
   - optionally project velocity targets for geometry modes.
6. Modify `src/train_step.py` only if needed to use the geometry velocity target and log diagnostics.
7. Add `configs/training_configs/train_owt_MODULUS-ELF-B-hybrid.yml`.
8. Run smoke checks with `latent_geometry=off` and `latent_geometry=hybrid_sphere`.
9. Do not promote any MODULUS result until the baseline and geometry-off gates clear.

## Constraints

- Do not change the tokenizer.
- Do not change the frozen T5 encoder.
- Do not change model size.
- Do not change the evaluator.
- Do not change the OpenWebText split.
- Do not change sampling configs for the first comparison.
- Do not claim improvement without matched baseline evidence.

## Preferred PR title

`Add MODULUS-ELF JAX overlay scaffold`

## Validation expected in PR body

- Baseline ELF-B eval status.
- Geometry-off smoke status.
- Hybrid-sphere smoke status.
- Known blockers.
