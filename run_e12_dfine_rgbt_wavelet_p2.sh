#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=/root/miniconda3/envs/affect/bin/python
CONFIG=configs/dfine/dfine_l_rgbt_wavelet_p2_aic.yml
E10_DIR=runs/e10_dfine_l_rgbt
OUTPUT=runs/e12_dfine_l_rgbt_wavelet_p2

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || {
  echo "output already exists: $OUTPUT" >&2
  echo "Refusing to overwrite an existing or interrupted experiment." >&2
  exit 1
}

if [[ -s "$E10_DIR/best_stg2.pth" ]]; then
  CHECKPOINT="$E10_DIR/best_stg2.pth"
elif [[ -s "$E10_DIR/best_stg1.pth" ]]; then
  CHECKPOINT="$E10_DIR/best_stg1.pth"
else
  echo "E10 checkpoint not found under $E10_DIR" >&2
  exit 1
fi

"$PYTHON" prepare_dfine_data.py \
  --output-dir data/prepared/dfine_e12 \
  --repeat-power 0.35 \
  --max-repeat 4

exec "$PYTHON" third_party/D-FINE/train.py \
  -c "$CONFIG" \
  --seed 3407 \
  -t "$CHECKPOINT"
