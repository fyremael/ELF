#!/usr/bin/env bash
set -euo pipefail

# Copy MODULUS-ELF overlay files into a local checkout of the upstream ELF repo.
# Usage:
#   scripts/apply_overlay_to_upstream.sh /path/to/upstream/ELF

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/upstream/ELF" >&2
  exit 2
fi

UPSTREAM_DIR="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${UPSTREAM_DIR}/src" ]]; then
  echo "error: ${UPSTREAM_DIR} does not look like an ELF checkout" >&2
  exit 1
fi

mkdir -p "${UPSTREAM_DIR}/src/utils"
mkdir -p "${UPSTREAM_DIR}/src/configs/training_configs"

cp "${REPO_ROOT}/overlays/src/utils/modulus_geometry.py" \
   "${UPSTREAM_DIR}/src/utils/modulus_geometry.py"

cp "${REPO_ROOT}/overlays/configs/training_configs/train_owt_MODULUS-ELF-B-hybrid.yml" \
   "${UPSTREAM_DIR}/src/configs/training_configs/train_owt_MODULUS-ELF-B-hybrid.yml"

cat <<'MSG'
Overlay files copied.

Manual integration still required for WP00:
  - add Config fields from docs/PATCHPOINTS.md
  - import modulus_geometry in src/utils/sampling_utils.py
  - gate behavior on config.latent_geometry
  - preserve exact upstream behavior for latent_geometry=off
MSG
