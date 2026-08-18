#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=/root/miniconda3/envs/affect/bin/python
CONFIG=configs/dfine/dfine_l_rgbtd_e13_aic.yml
E12_DIR=runs/e12_dfine_l_rgbt_wavelet_p2
OUTPUT=runs/e13_dfine_l_rgbtd_balanced

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || {
  echo "output already exists: $OUTPUT" >&2
  echo "Refusing to overwrite an existing or interrupted experiment." >&2
  exit 1
}

if [[ -s "$E12_DIR/best_stg2.pth" ]]; then
  CHECKPOINT="$E12_DIR/best_stg2.pth"
elif [[ -s "$E12_DIR/best_stg1.pth" ]]; then
  CHECKPOINT="$E12_DIR/best_stg1.pth"
else
  echo "E12 checkpoint not found under $E12_DIR" >&2
  exit 1
fi

"$PYTHON" prepare_e13_data.py \
  --repeat-power 0.35 \
  --max-repeat 4 \
  --seed 3407

exec "$PYTHON" third_party/D-FINE/train.py \
  -c "$CONFIG" \
  --seed 3407 \
  -t "$CHECKPOINT"
