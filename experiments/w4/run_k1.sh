#!/usr/bin/env bash
# W4 ② K1 memory-tight workload (docs/plans/W4_PLAN.md §4). Same KV byte budget for both models,
# 2 images (896x896) + 64 text tokens in, 256 out (ignore-eos), closed-loop concurrency.
# /metrics sampled every 1 s per point (running / waiting / KV usage / preemptions).
# CALIBRATE=1: B0 only, one concurrency, one rep (§4.2).
# REP_START=n: start at rep n (re-running invalidated points into a separate run dir).
# SIDE (default 896) / KEEP (default 0.5): image side and keep ratio (follow-up S1, 2026-10-24).
# OUT (default 256): output tokens of the measured points (follow-up T5).
# Usage (from the repo root): KV_BYTES=4294967296 bash experiments/w4/run_k1.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
MODELS=(${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B})
if [ "${CALIBRATE:-0}" = 1 ]; then
  METHODS=(${METHODS:-none}); CONCS=(${CONCS:-128}); REPS=${REPS:-1}
else
  METHODS=(${METHODS:-none downsample random rate divprune}); CONCS=(${CONCS:-32 64 128 256}); REPS=${REPS:-3}
fi
KV_BYTES=${KV_BYTES:?KV cache bytes}
PORT=${PORT:-8071}
SIDE=${SIDE:-896}
KEEP=${KEEP:-0.5}
IMGCFG="{($SIDE, $SIDE, 1): 1.0}"
LIMIT='{"image": 2, "video": 0}'
mkdir -p "$RUN/results"
$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || exit 5
echo "[$(date -Is)] K1: side=$SIDE keep=$KEEP out=${OUT:-256} calibrate=${CALIBRATE:-0} kv_bytes=$KV_BYTES max_num_seqs=${MAX_SEQS:-128} models=${MODELS[*]} methods=${METHODS[*]} concs=${CONCS[*]} reps=${REP_START:-1}..$REPS"

bench() {  # bench <model> <out_len> <num_prompts> <concurrency> <seed> [extra args...]
  local m=$1 out=$2 n=$3 c=$4 seed=$5; shift 5
  "$ROOT/envs/vts/bin/vllm" bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "http://127.0.0.1:$PORT" --model "$m" --tokenizer "$ROOT/models/$m" \
    --dataset-name random-mm --random-input-len 64 --random-output-len "$out" --random-range-ratio 0 \
    --random-mm-base-items-per-request 2 --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt "$LIMIT" --random-mm-bucket-config "$IMGCFG" \
    --ignore-eos --temperature 0 --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
    --num-prompts "$n" --max-concurrency "$c" --request-rate inf --seed "$seed" "$@"
}

STATUS=0
for rep in $(seq "${REP_START:-1}" "$REPS"); do
  for METHOD in "${METHODS[@]}"; do
    for M in "${MODELS[@]}"; do
      if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
        echo "[$(date -Is)] GPU busy before rep=$rep $METHOD $M, abort"; exit 4
      fi
      if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy"; exit 6; fi
      if [ "$METHOD" = none ]; then unset VTS_METHOD VTS_KEEP; else export VTS_METHOD=$METHOD VTS_KEEP=$KEEP VTS_SEED=0; fi
      LOG=$RUN/serve_${M}_${METHOD}_rep${rep}.log
      "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" --served-model-name "$M" \
        --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --gpu-memory-utilization 0.90 \
        --kv-cache-memory-bytes "$KV_BYTES" --max-model-len 4096 --max-num-seqs "${MAX_SEQS:-128}" \
        --limit-mm-per-prompt "$LIMIT" --seed 0 > "$LOG" 2>&1 &
      SP=$!
      ok=0
      for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
        kill -0 $SP 2>/dev/null || break
        sleep 5
      done
      if [ $ok -ne 1 ]; then
        echo "[$(date -Is)] serve $M/$METHOD failed"; grep -E "Error" "$LOG" | tail -5
        kill $SP 2>/dev/null; wait $SP 2>/dev/null; STATUS=1; continue
      fi
      echo "[$(date -Is)] rep=$rep $M/$METHOD healthy; $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$LOG" | tail -1); plugin lines $(grep -c vts_plugin "$LOG")"
      # discarded warm-up with its own seed (no cache sharing with measured points)
      bench "$M" 32 16 8 999999 > "$RUN/warmup_${M}_${METHOD}_rep${rep}.log" 2>&1
      for C in "${CONCS[@]}"; do
        N=$(( C * 4 )); [ $N -lt 256 ] && N=256
        SEED=$(( 300000 + rep * 10000 + C ))   # unique per point, shared across models and methods (paired)
        TAG=${M}_${METHOD}_c${C}_rep${rep}
        STOP=$RUN/results/$TAG.stop; rm -f "$STOP"
        $PY experiments/w4/metrics_sampler.py --url "http://127.0.0.1:$PORT/metrics" \
          --out "$RUN/results/$TAG.metrics.csv" --stop-file "$STOP" &
        MS=$!
        bench "$M" "${OUT:-256}" "$N" "$C" "$SEED" --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
          --save-result --result-dir "$RUN/results" --result-filename "$TAG.json" > "$RUN/results/$TAG.log" 2>&1
        rc=$?
        touch "$STOP"; wait $MS 2>/dev/null
        [ $rc -eq 0 ] || { echo "bench $TAG rc=$rc"; tail -15 "$RUN/results/$TAG.log"; STATUS=1; }
        echo "$TAG: $(grep -E 'Request throughput|Failed requests|Output token throughput' "$RUN/results/$TAG.log" | tr -s ' ' | tr '\n' ';') $($PY experiments/w4/k1_steady.py "$RUN/results/$TAG.metrics.csv")"
      done
      echo "cache check rep=$rep $M/$METHOD: $(grep -o 'Prefix cache hit rate: [0-9.]*%, MM cache hit rate: [0-9.]*%' "$LOG" | tail -1); oom_or_dead=$(grep -c -E 'OutOfMemoryError|EngineDeadError' "$LOG")"
      kill $SP; wait $SP 2>/dev/null
    done
  done
done
unset VTS_METHOD VTS_KEEP
echo "[$(date -Is)] K1_DONE status=$STATUS"
exit $STATUS
