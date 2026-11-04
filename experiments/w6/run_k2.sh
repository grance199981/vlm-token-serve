#!/usr/bin/env bash
# W4.5 K2 multi-turn image chat (docs/plans/K2_PLAN.md). Same KV byte budget as K1, prefix caching on,
# --enable-prompt-tokens-details; per point S fresh sessions x 4 turns, round-robin, 8 in flight
# (experiments/w6/k2_client.py). /metrics sampled every 1 s per point.
# CALIBRATE=1: B0 only, one rep. SMOKE=1: Qwen3.5 B0, S = 8 16, one rep.
# No plugin reinstall (0.1.7 is installed; the version is checked against the release).
# Usage (from the repo root): bash experiments/w6/run_k2.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
if [ "${SMOKE:-0}" = 1 ]; then
  MODELS=(Qwen3.5-4B); METHODS=(none); SESS=(8 16); REPS=1
elif [ "${CALIBRATE:-0}" = 1 ]; then
  MODELS=(Qwen3-VL-4B-Instruct Qwen3.5-4B); METHODS=(none); SESS=(8 16 24 32 48 64 96 128); REPS=1
else
  MODELS=(Qwen3-VL-4B-Instruct Qwen3.5-4B); METHODS=(${METHODS:-none downsample random})
  SESS=(${SESS:-8 16 20 24 28 32 36 40 44 48 56 64 72 80 96}); REPS=${REPS:-2}   # finer grid, set after calibration (K2_PLAN §3)
fi
KV_BYTES=${KV_BYTES:-4294967296}
PORT=${PORT:-8096}
KEEP=${KEEP:-0.5}
LIMIT='{"image": 1, "video": 0}'
mkdir -p "$RUN/results"
ver=$($PY -c "import vts_plugin; print(vts_plugin.__version__)" 2>/dev/null)
want=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
[ "$ver" = "$want" ] || { echo "installed vts_plugin $ver != release $want"; exit 5; }
echo "[$(date -Is)] K2: smoke=${SMOKE:-0} calibrate=${CALIBRATE:-0} kv_bytes=$KV_BYTES keep=$KEEP models=${MODELS[*]} methods=${METHODS[*]} sessions=${SESS[*]} reps=$REPS plugin=$ver"

STATUS=0
for rep in $(seq 1 "$REPS"); do
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
        --kv-cache-memory-bytes "$KV_BYTES" --max-model-len 4096 --max-num-seqs 128 \
        --limit-mm-per-prompt "$LIMIT" --enable-prompt-tokens-details --seed 0 > "$LOG" 2>&1 &
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
      # discarded warm-up with its own seed (its sessions are never revisited)
      $PY experiments/w6/k2_client.py --url "http://127.0.0.1:$PORT" --model "$M" --sessions 4 --turns 2 \
        --seed 999 --out "$RUN/warmup_${M}_${METHOD}_rep${rep}.jsonl" > "$RUN/warmup_${M}_${METHOD}_rep${rep}.log" 2>&1
      for S in "${SESS[@]}"; do
        SEED=$(( 500000 + rep * 10000 + S ))   # fresh sessions per point, shared across models and methods (paired)
        TAG=${M}_${METHOD}_s${S}_rep${rep}
        STOP=$RUN/results/$TAG.stop; rm -f "$STOP"
        $PY experiments/w4/metrics_sampler.py --url "http://127.0.0.1:$PORT/metrics" \
          --out "$RUN/results/$TAG.metrics.csv" --stop-file "$STOP" &
        MS=$!
        t0=$(date +%s)
        $PY experiments/w6/k2_client.py --url "http://127.0.0.1:$PORT" --model "$M" --sessions "$S" \
          --seed "$SEED" --out "$RUN/results/$TAG.jsonl" > "$RUN/results/$TAG.log" 2>&1
        rc=$?
        touch "$STOP"; wait $MS 2>/dev/null
        [ $rc -eq 0 ] || { echo "client $TAG rc=$rc"; tail -15 "$RUN/results/$TAG.log"; STATUS=1; }
        echo "$TAG: $(( $(date +%s) - t0 )) s; $($PY - "$RUN/results/$TAG.jsonl" <<'EOF'
import json, sys
r = [json.loads(x) for x in open(sys.argv[1])]
l = [x for x in r if x["turn"] >= 2 and x["prompt_tokens"]]
e = sum(x["error"] is not None for x in r)
h = sum(x["cached_tokens"] or 0 for x in l) / max(1, sum(x["prompt_tokens"] for x in l))
print(f"requests={len(r)} errors={e} hit_t2-4={100 * h:.1f}%")
EOF
)"
      done
      echo "cache check rep=$rep $M/$METHOD: $(grep -o 'Prefix cache hit rate: [0-9.]*%, MM cache hit rate: [0-9.]*%' "$LOG" | tail -1); oom_or_dead=$(grep -c -E 'OutOfMemoryError|EngineDeadError' "$LOG")"
      kill $SP; wait $SP 2>/dev/null
    done
  done
done
unset VTS_METHOD VTS_KEEP
echo "[$(date -Is)] K2_DONE status=$STATUS"
exit $STATUS
