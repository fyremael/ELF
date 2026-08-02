#!/usr/bin/env bash
set -euo pipefail

# Run the first MODULUS-ELF hybrid-sphere training config after manual integration.
# Run inside an upstream ELF checkout after applying and wiring the overlay.

cd "${ELF_SRC_DIR:-src}"

python train.py \
  --config configs/training_configs/train_owt_MODULUS-ELF-B-hybrid.yml
