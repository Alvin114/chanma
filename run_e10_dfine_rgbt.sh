#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8

PYTHON=/root/miniconda3/envs/affect/bin/python
CHECKPOINT=weights/dfine_l_obj2coco_e25.pth
CHECKPOINT_URL=https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth
OUTPUT=runs/e10_dfine_l_rgbt

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 1; }

"$PYTHON" prepare_dfine_data.py
mkdir -p weights
if [[ ! -s "$CHECKPOINT" ]]; then
  curl -fL --retry 5 "$CHECKPOINT_URL" -o "$CHECKPOINT"
fi

exec "$PYTHON" third_party/D-FINE/train.py \
  -c configs/dfine/dfine_l_rgbt_aic.yml \
  --seed 3407 \
  -t "$CHECKPOINT"
