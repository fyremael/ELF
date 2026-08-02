# MODULUS-ELF

Grand Challenge Labs working repository for **MODULUS-ELF**: a governed experimental overlay around the official JAX implementation of **ELF: Embedded Language Flows**.

This repository is intentionally initialized as an overlay repository rather than a vendored copy. The upstream ELF codebase remains the source baseline.

- Upstream ELF: https://github.com/lillian039/ELF
- Paper: https://arxiv.org/abs/2605.10938
- Initial branch: `agent/modulus-elf-jax-overlay`

## First governed objective

Reproduce the released ELF-B OpenWebText checkpoint evaluation before introducing MODULUS geometry changes.

The baseline gate is:

```text
python eval.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path embedded-language-flows/ELF-B-owt
```

Expected sanity target from upstream documentation:

```text
ELF-B, OpenWebText, 32-step SDE: Gen. PPL approximately 24, entropy approximately 5.15
```

## MODULUS intervention discipline

The first MODULUS branch must be minimal and falsifiable:

1. Preserve upstream data, tokenizer, frozen T5 encoder, model size, sampling schedule, evaluator, and checkpoint/evaluation protocol.
2. Add latent-geometry controls around the flow-matching latent interface.
3. Verify that `latent_geometry=off` remains behavior-preserving.
4. Only then train/evaluate the first `hybrid_sphere` variant.

## Repository layout

```text
docs/       governed work package, patch map, acceptance rubric
scripts/    local reproduction and overlay setup commands
overlays/   proposed JAX source/config additions
codex/      implementation prompt for a coding agent
```

This repository does not yet claim a MODULUS result. It records the reproducibility and intervention plan required to test one.
