#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8
AFFECT_PYTHON="${AFFECT_PYTHON:-/root/miniconda3/envs/affect/bin/python}"
if [[ ! -x "$AFFECT_PYTHON" ]]; then
  echo "Affect Python not found or not executable: $AFFECT_PYTHON" >&2
  echo "Override it with: AFFECT_PYTHON=/path/to/affect/bin/python $0" >&2
  exit 1
fi


for run_dir in runs/e07_ir runs/e08_rgbt_m2d_lif; do
  if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to reuse non-empty run directory: $run_dir" >&2
    exit 1
  fi
done

"$AFFECT_PYTHON" train.py --config configs/ir.yaml
"$AFFECT_PYTHON" train.py --config configs/rgbt_m2d_lif.yaml
