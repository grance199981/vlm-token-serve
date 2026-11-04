#!/usr/bin/env bash
# T5b: 100 single-image requests, vLLM (vanilla_nopc, random50_nopc) vs HF reference
# (keep_all, random50, random50 with placeholder positions as a diagnostic).
# Usage (from the repo root): bash experiments/w2/run_t5b.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
PORT=8051
LE=1003520
mkdir -p "$RUN/out"
SET="--n-chart 50 --n-text 50 --single-only --max-tokens 16"
$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || exit 5
STATUS=0
for M in ${MODELS:-Qwen3.5-4B Qwen3-VL-4B-Instruct}; do
  for C in "vanilla_nopc|" "random50_nopc|VTS_METHOD=random VTS_KEEP=0.5"; do
    IFS='|' read -r NAME ENVS <<< "$C"
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
      echo "GPU busy"; exit 4
    fi
    env -u VTS_METHOD -u VTS_KEEP $ENVS "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" --served-model-name "$M" \
      --host 127.0.0.1 --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 16384 \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' --no-enable-prefix-caching \
      --mm-processor-kwargs "{\"size\": {\"longest_edge\": $LE, \"shortest_edge\": 65536}}" --seed 0 \
      > "$RUN/serve_${M}_${NAME}.log" 2>&1 &
    SP=$!
    for _ in $(seq 1 240); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
      kill -0 $SP 2>/dev/null || break
      sleep 5
    done
    $PY experiments/w2/integration_client.py --base-url "http://127.0.0.1:$PORT" --model "$M" \
      --model-dir "$ROOT/models/$M" --out "$RUN/out/${M}_${NAME}.json" --longest-edge $LE --passes 1 $SET \
      > "$RUN/client_${M}_${NAME}.log" 2>&1
    echo "[$(date -Is)] $M vllm $NAME rc=$?"
    kill $SP
    wait $SP 2>/dev/null
  done
  for spec in "keep_all 1.0 hf_keep_all original" "random 0.5 hf_random50 original" \
              "random 0.5 hf_random50_placeholder placeholder"; do
    set -- $spec
    $PY experiments/w2/hf_prune_reference.py --model-dir "$ROOT/models/$M" --method "$1" --keep "$2" \
      --pos-mode "$4" --out "$RUN/out/${M}_$3.json" $SET > "$RUN/hf_${M}_$3.log" 2>&1
    echo "[$(date -Is)] $M $3 rc=$?"
  done
  $PY experiments/w2/t5b_compare.py "$RUN/out" "$RUN/out" "$M" | tee -a "$RUN/t5b.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || STATUS=1
done
echo "[$(date -Is)] T5B_DONE status=$STATUS"
exit $STATUS
