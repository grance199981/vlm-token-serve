#!/usr/bin/env bash
# Route C (docs/W3_CONDITION_ANALYSIS.md §12): ST-VQA M4C subval + InfoVQA val,
# 2 models x {B0, downsample, rate} x keep {0.25, 0.125}; server settings identical to W3.
# One server per config, both tasks evaluated on it. Does NOT reinstall the plugin (K1 shares
# the env); checks the installed version. SMOKE=1: one config, 10 samples per task.
# Usage (from the repo root): bash experiments/routec/run_routec.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
PORT=${PORT:-8092}
MODELS=(${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B})
CONFIGS=("b0|none 1.0" "downsample_k25|downsample 0.25" "rate_k25|rate 0.25" "downsample_k125|downsample 0.125" "rate_k125|rate 0.125")
LIMIT=""
if [ "${SMOKE:-0}" = 1 ]; then MODELS=(Qwen3.5-4B); CONFIGS=("rate_k25|rate 0.25"); LIMIT="--limit 10"; fi
mkdir -p "$RUN"
ver=$($PY -c "import vts_plugin; print(vts_plugin.__version__)" 2>/dev/null)
want=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)  # the release's own version
[ "$ver" = "$want" ] || { echo "installed vts_plugin is '$ver', expected $want"; exit 5; }
$PY experiments/routec/prepare_routec.py --root "$ROOT" > "$RUN/prepare_routec.log" 2>&1 || { tail -20 "$RUN/prepare_routec.log"; exit 5; }
echo "[$(date -Is)] $(tail -1 "$RUN/prepare_routec.log")"
echo "[$(date -Is)] models=${MODELS[*]} configs=${#CONFIGS[@]} smoke=${SMOKE:-0} gpu=$CUDA_VISIBLE_DEVICES plugin=$ver"
STATUS=0
for M in "${MODELS[@]}"; do
  for C in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME SPEC <<< "$C"
    read -r METHOD KEEP <<< "$SPEC"
    OUT=$RUN/$M/$NAME
    mkdir -p "$OUT"
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
      echo "[$(date -Is)] GPU busy before $M/$NAME, abort"; exit 4
    fi
    if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy"; exit 6; fi
    if [ "$METHOD" = none ]; then unset VTS_METHOD VTS_KEEP; else export VTS_METHOD=$METHOD VTS_KEEP=$KEEP VTS_SEED=0; fi
    "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" \
      --served-model-name "$M" --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
      --gpu-memory-utilization 0.90 --max-model-len 32768 --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' --seed 0 > "$OUT/serve.log" 2>&1 &
    SP=$!
    unset VTS_METHOD VTS_KEEP
    ok=0
    for _ in $(seq 1 240); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
      kill -0 $SP 2>/dev/null || break
      sleep 5
    done
    if [ $ok -ne 1 ]; then
      echo "[$(date -Is)] $M/$NAME server failed"; grep -E "Error" "$OUT/serve.log" | tail -5
      kill $SP 2>/dev/null; wait $SP 2>/dev/null; STATUS=1; continue
    fi
    MA="model=$M,base_url=http://127.0.0.1:$PORT/v1,api_key=EMPTY,num_concurrent=32,timeout=600,max_size_in_mb=64,enable_thinking_kwarg=False,httpx_trust_env=False"
    for TASK in vts_stvqa_subval vts_infovqa_val; do
      t0=$(date +%s)
      OPENAI_API_KEY=EMPTY "$ROOT/envs/vts/bin/lmms-eval" eval --model openai --model_args "$MA" \
        --tasks "$TASK" --include_path "$ROOT/tasks_routec" $LIMIT --batch_size 1 --seed 0 \
        --log_samples --output_path "$OUT/lmms_eval" >> "$OUT/lmms_eval_$TASK.log" 2>&1 || { echo "lmms-eval $TASK failed"; STATUS=1; }
      echo "[$(date -Is)]   $M/$NAME $TASK in $(( $(date +%s) - t0 ))s, failed_requests=$(grep -c 'All 5 attempts failed' "$OUT/lmms_eval_$TASK.log")"
    done
    nfail=$(cat "$OUT"/lmms_eval_*.log | grep -c "All 5 attempts failed")
    noom=$(grep -c -E "OutOfMemoryError|EngineDeadError" "$OUT/serve.log")
    valid=yes
    if [ "$nfail" -gt 0 ] || [ "$noom" -gt 0 ]; then valid=NO; STATUS=1; fi
    echo "$valid" > "$OUT/valid.txt"
    echo "[$(date -Is)] $M/$NAME ($METHOD keep=$KEEP) plugin lines $(grep -c vts_plugin "$OUT/serve.log"), failed_requests=$nfail oom_or_dead=$noom valid=$valid"
    kill $SP; wait $SP 2>/dev/null
  done
done
echo "[$(date -Is)] ROUTEC_DONE status=$STATUS"
exit $STATUS
