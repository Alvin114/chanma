#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8

PYTHON=/root/miniconda3/envs/affect/bin/python
E10_DIR=runs/e10_dfine_l_rgbt
OUTPUT=runs/e10f_dfine_l_rgbt_all

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 1; }

if [[ -s "$E10_DIR/best_stg2.pth" ]]; then
  CHECKPOINT="$E10_DIR/best_stg2.pth"
elif [[ -s "$E10_DIR/best_stg1.pth" ]]; then
  CHECKPOINT="$E10_DIR/best_stg1.pth"
else
  echo "E10 checkpoint not found under $E10_DIR" >&2
  exit 1
fi

"$PYTHON" prepare_dfine_data.py

exec "$PYTHON" third_party/D-FINE/train.py \
  -c configs/dfine/dfine_l_rgbt_all_aic.yml \
  --seed 3407 \
  -t "$CHECKPOINT"
