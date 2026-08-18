#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON=/root/miniconda3/envs/affect/bin/python
CONFIG=configs/dfine/dfine_l_rgbtd_e13_aic.yml
OUTPUT=runs/e13_dfine_l_rgbtd_balanced
SUBMISSION_DIR="$OUTPUT/submission_txt"
SUBMISSION_ZIP="$OUTPUT/submission.zip"

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || {
  echo "output already exists: $OUTPUT" >&2
  echo "Refusing to overwrite an existing or interrupted experiment." >&2
  exit 1
}

./run_e13_dfine_rgbtd_balanced.sh

CHECKPOINT="$("$PYTHON" select_best_dfine_checkpoint.py "$OUTPUT")"
echo "selected checkpoint: $CHECKPOINT"

"$PYTHON" predict_dfine.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --confidence 0.01 \
  --output-dir "$SUBMISSION_DIR" \
  --zip "$SUBMISSION_ZIP"

echo "checkpoint: $CHECKPOINT"
echo "upload: $SUBMISSION_ZIP"
