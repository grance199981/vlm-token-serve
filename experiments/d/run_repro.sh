#!/usr/bin/env bash
# Repro for vllm-project/vllm#63187: stock vLLM video pruning vs control.
# Usage (from the repo root): bash experiments/d/run_repro.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
MODEL=$ROOT/models/Qwen3-VL-4B-Instruct
mkdir -p "$RUN"
unset VTS_METHOD
STATUS=0
# name | pruning rate | max_num_batched_tokens   (control first)
for SPEC in "d2_noprune_b16384|0|16384" "d1_evs075_b16384|0.75|16384"             "d2_noprune_b2048|0|2048" "d1_evs075_b2048|0.75|2048"; do
  IFS='|' read -r NAME RATE BT <<< "$SPEC"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
    echo "[$(date -Is)] GPU busy before $NAME, abort"; exit 4
  fi
  echo "[$(date -Is)] start $NAME"
  timeout 1800 $PY experiments/d/encoder_budget_repro.py --model "$MODEL" --out "$RUN/$NAME.json"     --pruning-rate "$RATE" --max-num-batched-tokens "$BT" > "$RUN/$NAME.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && STATUS=1
  echo "[$(date -Is)] $NAME rc=$rc $(tail -1 "$RUN/$NAME.log" | cut -c1-400)"
done
echo "[$(date -Is)] REPRO_DONE status=$STATUS"
exit $STATUS
