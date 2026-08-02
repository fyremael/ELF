# MODULUS-ELF Run Matrix

## Stage A: upstream baseline

| Run | Source | Config | Checkpoint | Geometry | Required status |
| --- | --- | --- | --- | --- | --- |
| A0 | `lillian039/ELF` | `train_owt_ELF-B.yml` | `ELF-B-owt` | off/upstream | must complete |

## Stage B: overlay preservation

| Run | Source | Config | Checkpoint | Geometry | Required status |
| --- | --- | --- | --- | --- | --- |
| B0 | upstream + overlay | `train_owt_ELF-B.yml` plus `latent_geometry=off` | `ELF-B-owt` | off | behavior-preserving smoke |

## Stage C: first MODULUS candidate

| Run | Source | Config | Checkpoint | Geometry | Required status |
| --- | --- | --- | --- | --- | --- |
| C0 | upstream + overlay | `train_owt_MODULUS-ELF-B-hybrid.yml` | train from scratch | hybrid_sphere | train/eval only after A0+B0 |

## Comparison rule

Only compare C0 to A0 when model size, dataset, tokenizer, frozen encoder, sampling schedule, generation sample count, PPL evaluator, and seed policy are recorded.
