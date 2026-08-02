# MODULUS-ELF-WP00: OpenWebText Baseline and Geometry Overlay

## Status

Candidate work package. No result claim is admitted.

## Objective

Establish a reproducible path from the official ELF JAX implementation to a first MODULUS-ELF experimental branch.

The work package has two gates:

1. **Baseline gate:** reproduce the official ELF-B OpenWebText checkpoint evaluation under the upstream evaluator.
2. **Intervention gate:** add geometry-controlled latent-flow machinery while preserving a behavior-equivalent `latent_geometry=off` mode.

No MODULUS geometry result may be promoted before the baseline gate clears.

## Upstream source of truth

- Repository: `lillian039/ELF`
- Paper: `arXiv:2605.10938`
- Baseline config: `src/configs/training_configs/train_owt_ELF-B.yml`
- Baseline checkpoint: `embedded-language-flows/ELF-B-owt`

## Baseline command

Run inside the upstream repository's `src/` directory:

```bash
python eval.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-owt
```

Expected sanity target from upstream documentation:

```text
ELF-B, OpenWebText, 32-step SDE: Gen. PPL approximately 24; entropy approximately 5.15.
```

Small metric drift is acceptable across hardware; a large deviation blocks MODULUS interpretation.

## Minimal intervention

The first MODULUS variant should not rewrite the ELF model. It should modify the latent flow interface only:

- encode text to `x0` exactly as upstream does;
- construct geometry-aware noisy states from `x0`, noise, and time `t`;
- project model outputs and velocity targets consistently;
- preserve conditioning-token behavior;
- leave decoding and evaluation unchanged.

## First geometry family

The first admissible intervention is `hybrid_sphere`:

- preserve tokenwise latent radius statistics;
- normalize only angular direction;
- use tangent-projected noise and velocity targets;
- avoid full unit-sphere clamping until hybrid results are stable.

Rationale: upstream ELF scales frozen T5 latents with `latent_std: 0.2` and uses comparatively large denoising/decoder noise scales. A full unit-sphere clamp may erase radius information too early.

## Acceptance criteria

Baseline accepted only if:

- the exact upstream checkpoint loads;
- the evaluator completes generation and PPL/entropy evaluation;
- output metrics are plausibly concordant with upstream sanity values;
- hardware, seed, sampler, config overrides, and output files are recorded.

MODULUS overlay accepted only if:

- `latent_geometry=off` remains behavior-preserving relative to upstream within expected nondeterminism;
- geometry diagnostics report radius, norm, cosine, and tangency statistics;
- training/evaluation commands are recorded;
- no OpenWebText result is promoted without matched baseline comparison.

## Non-claims

This work package does not claim:

- improved OpenWebText perplexity;
- improved language modeling quality;
- a replacement for ELF;
- a general result about diffusion language models;
- a production-ready implementation.

It only establishes the governed path to test those hypotheses.
