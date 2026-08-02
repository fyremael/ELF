#!/usr/bin/env bash
set -euo pipefail

# Reproduce the official ELF-B OpenWebText checkpoint evaluation.
# Run from a machine with the required JAX/TPU or compatible accelerator setup.

UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/lillian039/ELF.git}"
WORKDIR="${WORKDIR:-./upstream_elf}"
CHECKPOINT="${CHECKPOINT:-embedded-language-flows/ELF-B-owt}"
CONFIG="${CONFIG:-configs/training_configs/train_owt_ELF-B.yml}"
SEED="${SEED:-42}"

if [[ ! -d "${WORKDIR}/.git" ]]; then
  git clone "${UPSTREAM_REPO}" "${WORKDIR}"
fi

cd "${WORKDIR}"
git rev-parse HEAD
pip install -r requirements.txt
cd src

python eval.py \
  --config "${CONFIG}" \
  --checkpoint_path "${CHECKPOINT}" \
  --seed "${SEED}"
