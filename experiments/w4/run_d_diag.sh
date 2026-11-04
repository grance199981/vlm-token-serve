#!/usr/bin/env bash
# W4 ⑥b (W4_RESULTS §4): ① D-shape request (1 image, 822 prompt tokens), B0, prefix caching on / off,
# both models; records the number of LLM forward calls and tokens per call. Plugin inactive, no reinstall.
# Usage (from the repo root): bash experiments/w4/run_d_diag.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
MODELS=(${MODELS:-Qwen3.5-4B Qwen3-VL-4B-Instruct})
mkdir -p "$RUN"
unset VTS_METHOD VTS_KEEP
echo "[$(date -Is)] D diag: models=${MODELS[*]} gpu=$CUDA_VISIBLE_DEVICES"
STATUS=0
for M in "${MODELS[@]}"; do
  for PC in on off; do
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
      echo "[$(date -Is)] GPU busy before $M/pc_$PC, abort"; exit 4
    fi
    timeout 1800 $PY experiments/w4/stage_bench.py --model "$ROOT/models/$M" --shapes D --prefix-caching "$PC" \
      --runs "${RUNS:-20}" --out "$RUN/$M/pc_$PC.json" > "$RUN/${M}_pc_$PC.log" 2>&1
    rc=$?; [ $rc -ne 0 ] && STATUS=1
    echo "[$(date -Is)] $M/pc_$PC rc=$rc $(grep -E '^D ' "$RUN/${M}_pc_$PC.log" | cut -c1-400)"
  done
done
echo "[$(date -Is)] DIAG_DONE status=$STATUS"
exit $STATUS
