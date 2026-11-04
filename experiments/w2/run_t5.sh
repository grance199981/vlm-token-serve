#!/usr/bin/env bash
# T5: HF reference for pruning vs the vLLM plugin outputs of integration rev2.
# Usage (from the repo root): bash experiments/w2/run_t5.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
VLLM_OUT=${VLLM_OUT:?set VLLM_OUT to the out/ dir of an integration run}
mkdir -p "$RUN/out"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then echo "GPU busy"; exit 4; fi
STATUS=0
for M in ${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B}; do
  for spec in "keep_all 1.0 keep_all" "random 0.5 random50" "pool2x2 1.0 pool2x2"; do
    set -- $spec
    t0=$(date +%s)
    $PY experiments/w2/hf_prune_reference.py --model-dir "$ROOT/models/$M" --method "$1" --keep "$2" \
      --out "$RUN/out/${M}_hf_$3.json" > "$RUN/hf_${M}_$3.log" 2>&1
    rc=$?
    echo "[$(date -Is)] $M hf $3 rc=$rc $(( $(date +%s) - t0 ))s"
    [ $rc -eq 0 ] || { tail -20 "$RUN/hf_${M}_$3.log"; STATUS=1; }
  done
  $PY experiments/w2/compare_hf_vllm.py "$RUN/out" "$VLLM_OUT" "$M" | tee -a "$RUN/t5.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || STATUS=1
done
echo "[$(date -Is)] T5_DONE status=$STATUS"
exit $STATUS
