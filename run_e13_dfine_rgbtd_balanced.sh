#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=/root/miniconda3/envs/affect/bin/python
CONFIG=configs/dfine/dfine_l_rgbtd_e13_aic.yml
CHECKPOINT=weights/dfine_l_obj2coco_e25.pth
CHECKPOINT_URL=https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth
OUTPUT=runs/e13_dfine_l_rgbtd_balanced

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || {
  echo "output already exists: $OUTPUT" >&2
  echo "Refusing to overwrite an existing or interrupted experiment." >&2
  exit 1
}

"$PYTHON" prepare_e13_data.py \
  --repeat-power 0.35 \
  --max-repeat 4 \
  --seed 3407

mkdir -p weights
if [[ ! -s "$CHECKPOINT" ]]; then
  curl -fL --retry 5 "$CHECKPOINT_URL" -o "$CHECKPOINT"
fi

exec "$PYTHON" third_party/D-FINE/train.py \
  -c "$CONFIG" \
  --seed 3407 \
  -t "$CHECKPOINT"
