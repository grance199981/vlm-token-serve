#!/usr/bin/env bash
# W2 integration tests T1-T3 on a real vLLM server (GPU0, exclusive).
# Installs the plugin from the release dir into envs/vts, then for each model runs
# a set of server configurations and the same fixed requests; check_integration.py
# compares the outputs.
#
# Usage (from the repo root): bash experiments/w2/run_integration.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
PORT=${PORT:-8041}
LE=1003520
MODELS=(${MODELS:-Qwen3.5-4B Qwen3-VL-4B-Instruct})
mkdir -p "$RUN/out"
# reuse outputs of an earlier run (same commit of the client) when only adding configs
if [ -n "${PREV_OUT:-}" ]; then cp -n "$PREV_OUT"/*.json "$RUN/out/" && echo "copied previous outputs from $PREV_OUT"; fi

echo "[$(date -Is)] install plugin from the repo root $(pwd)"
$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || { tail -20 "$RUN/pip_install.log"; exit 5; }
$PY -c "import importlib.metadata as m; print('vts-plugin', m.version('vts-plugin')); print([e.value for e in m.entry_points(group='vllm.general_plugins') if e.name=='vts'])"

# name | env | extra serve args
CONFIGS=(
  "vanilla||"
  "vanilla_nopc||--no-enable-prefix-caching"
  "vanilla_c256||--max-num-batched-tokens 256"
  "keep_all|VTS_METHOD=keep_all|"
  "keep_all_nopc|VTS_METHOD=keep_all|--no-enable-prefix-caching"
  "random50|VTS_METHOD=random VTS_KEEP=0.5|"
  "random50_c256|VTS_METHOD=random VTS_KEEP=0.5|--max-num-batched-tokens 256"
  "random50_nopc|VTS_METHOD=random VTS_KEEP=0.5|--no-enable-prefix-caching"
  "random50_shift7|VTS_METHOD=random VTS_KEEP=0.5 VTS_DEBUG_POS_SHIFT=7|--no-enable-prefix-caching"
  "random50_transpose|VTS_METHOD=random VTS_KEEP=0.5 VTS_DEBUG_POS_TRANSPOSE=1|--no-enable-prefix-caching"
  "vanilla_evs_nopc||--no-enable-prefix-caching --video-pruning-rate 0.5"
  "divprune25|VTS_METHOD=divprune VTS_KEEP=0.25|"
  "rate25|VTS_METHOD=rate VTS_KEEP=0.25|"
  "rate_pr25|VTS_METHOD=rate_pr VTS_KEEP=0.25|"
  "pool2x2|VTS_METHOD=pool2x2|"
)

STATUS=0
for M in "${MODELS[@]}"; do
  for C in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME ENVS EXTRA <<< "$C"
    if [ -n "${ONLY:-}" ] && ! [[ " $ONLY " == *" $NAME "* ]]; then continue; fi
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
      echo "GPU busy before $M/$NAME, abort"; exit 4
    fi
    LOG=$RUN/serve_${M}_${NAME}.log
    env -u VTS_METHOD -u VTS_KEEP -u VTS_TAIL -u VTS_DEBUG_POS_SHIFT -u VTS_DEBUG_POS_TRANSPOSE $ENVS "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" --served-model-name "$M" \
      --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --gpu-memory-utilization 0.85 \
      --max-model-len 16384 --limit-mm-per-prompt '{"image": 2, "video": 0}' \
      --mm-processor-kwargs "{\"size\": {\"longest_edge\": $LE, \"shortest_edge\": 65536}}" \
      --seed 0 $EXTRA > "$LOG" 2>&1 &
    SP=$!
    ok=0
    for _ in $(seq 1 240); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
      kill -0 $SP 2>/dev/null || break
      sleep 5
    done
    if [ $ok -ne 1 ]; then echo "[$(date -Is)] $M/$NAME: server failed"; grep -E "Error|error" "$LOG" | tail -8; kill $SP 2>/dev/null; wait $SP 2>/dev/null; STATUS=1; continue; fi
    plugin_lines=$(grep -c "vts_plugin" "$LOG")
    $PY experiments/w2/integration_client.py --base-url "http://127.0.0.1:$PORT" --model "$M" \
      --model-dir "$ROOT/models/$M" --out "$RUN/out/${M}_${NAME}.json" --longest-edge $LE > "$RUN/client_${M}_${NAME}.log" 2>&1
    rc=$?
    echo "[$(date -Is)] $M/$NAME client rc=$rc vts_plugin_log_lines=$plugin_lines $(grep -o 'Prefix cache hit rate: [0-9.]*%' "$LOG" | tail -1)"
    [ $rc -eq 0 ] || { tail -15 "$RUN/client_${M}_${NAME}.log"; STATUS=1; }
    kill $SP; wait $SP 2>/dev/null
  done
done
$PY experiments/w2/check_integration.py "$RUN/out" | tee "$RUN/check.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || STATUS=1
echo "[$(date -Is)] W2_INTEGRATION_DONE status=$STATUS"
exit $STATUS
