#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=/root/miniconda3/envs/affect/bin/python
OUTPUT=runs/e10_dfine_l_rgbt
SUBMISSION_DIR="$OUTPUT/submission_txt"
SUBMISSION_ZIP="$OUTPUT/submission.zip"

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || {
  echo "output already exists: $OUTPUT" >&2
  echo "Refusing to overwrite an existing or interrupted experiment." >&2
  exit 1
}

./run_e10_dfine_rgbt.sh

if [[ -s "$OUTPUT/best_stg2.pth" ]]; then
  CHECKPOINT="$OUTPUT/best_stg2.pth"
elif [[ -s "$OUTPUT/best_stg1.pth" ]]; then
  CHECKPOINT="$OUTPUT/best_stg1.pth"
elif [[ -s "$OUTPUT/last.pth" ]]; then
  CHECKPOINT="$OUTPUT/last.pth"
else
  echo "training finished but no usable checkpoint was found under $OUTPUT" >&2
  exit 1
fi

echo "selected checkpoint: $CHECKPOINT"

"$PYTHON" predict_dfine.py \
  --config configs/dfine/dfine_l_rgbt_aic.yml \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --confidence 0.01 \
  --output-dir "$SUBMISSION_DIR" \
  --zip "$SUBMISSION_ZIP"

echo "checkpoint: $CHECKPOINT"
echo "upload: $SUBMISSION_ZIP"
