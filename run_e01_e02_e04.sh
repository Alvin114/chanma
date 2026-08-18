#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

CONDA_BIN="/root/miniconda3/bin/conda"
CONDA_ENV="affect"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi

for required_file in \
  data/train/AIC2026_Train_2000.zip \
  data/test/AIC2026_PHASE_1_1000.zip \
  data/prepared/manifests/train.jsonl \
  data/prepared/manifests/val.jsonl \
  weights/yolov5s_state.pt; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

run_experiment() {
  local experiment="$1"
  local config="$2"

  echo
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting ${experiment}: ${config}"
  "${CONDA_BIN}" run -n "${CONDA_ENV}" --no-capture-output \
    python train.py --config "${config}"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Finished ${experiment}"
}

run_experiment "E01 RGB" "configs/rgb.yaml"
run_experiment "E02 RGB-T ICAFusion" "configs/rgbt_icafusion.yaml"
run_experiment "E04 RGB-T-D" "configs/rgbtd.yaml"

echo
echo "All experiments completed: E01 -> E02 -> E04"
