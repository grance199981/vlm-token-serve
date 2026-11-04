#!/usr/bin/env bash
# W1 serving baseline (B0, no compression): `vllm bench serve` with random-mm images.
#   P = prefill-dominated: 4 images/request, 896x896, 64 text tokens in, 32 out
#   D = decode-dominated : 1 image/request,  896x896, 64 text tokens in, 512 out
# Concurrency sweep; repetitions alternate models A/B/A/B (server restarted each time).
# Every point uses its own seed so no request content repeats across points (cache hygiene).
# Timing setup: CUDA graphs on (no --enforce-eager), vLLM defaults for chunked prefill /
# prefix caching (recorded from the log), target GPU must be empty.
#
# Usage: bash experiments/w1/run_serving.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
MODELS=(${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B})
WORKLOADS=(${WORKLOADS:-P D})
CONCS=(${CONCS:-1 4 8 16 32 64})
REPS=${REPS:-3}
PORT=${PORT:-8031}
MEM_UTIL=${MEM_UTIL:-0.90}
IMG=${IMG:-896}
mkdir -p "$RUN/results"

smi() { nvidia-smi --query-gpu=uuid,memory.used,memory.total,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; }
gpu_busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; }

echo "[$(date -Is)] W1 serving: models=${MODELS[*]} workloads=${WORKLOADS[*]} concs=${CONCS[*]} reps=$REPS img=${IMG}x${IMG}"
echo "--- nvidia-smi before"; smi | tee "$RUN/nvidia_smi_before.txt"

STATUS=0
for rep in $(seq 1 "$REPS"); do
  for M in "${MODELS[@]}"; do
    if gpu_busy; then echo "[$(date -Is)] target GPU busy before rep=$rep $M, abort"; smi; exit 4; fi
    if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy"; exit 6; fi
    LOG=$RUN/serve_${M}_rep${rep}.log
    "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" --served-model-name "$M" \
      --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --gpu-memory-utilization "$MEM_UTIL" \
      --max-model-len 16384 --limit-mm-per-prompt '{"image": 4, "video": 0}' \
      --seed 0 > "$LOG" 2>&1 &
    SP=$!
    ok=0
    for _ in $(seq 1 240); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
      kill -0 $SP 2>/dev/null || break
      sleep 5
    done
    if [ $ok -ne 1 ]; then echo "serve $M failed"; tail -30 "$LOG"; kill $SP 2>/dev/null; STATUS=1; continue; fi
    echo "[$(date -Is)] rep=$rep $M healthy (pid $SP)"
    # Bottleneck probes: GPU SM utilisation every 1 s, CPU% of the API server and
    # EngineCore every 2 s (is the P-workload plateau GPU compute or CPU image decoding?)
    nvidia-smi dmon -i "$CUDA_VISIBLE_DEVICES" -s u -d 1 -o T > "$RUN/dmon_${M}_rep${rep}.log" 2>&1 &
    DMON=$!
    # instantaneous CPU% from /proc/<pid>/stat deltas (ps %cpu is a lifetime average)
    ( TCK=$(getconf CLK_TCK); EC=$(pgrep -P $SP -f EngineCore | head -1)
      ticks() { awk '{print $14 + $15}' /proc/$1/stat 2>/dev/null || echo 0; }
      a0=$(ticks $SP); e0=$(ticks $EC)
      echo "# time api_server_cpu% enginecore_cpu% (100 = one core)"
      while kill -0 $SP 2>/dev/null; do
        sleep 2
        a1=$(ticks $SP); e1=$(ticks $EC)
        echo "$(date +%H:%M:%S) $(( (a1 - a0) * 50 / TCK )) $(( (e1 - e0) * 50 / TCK ))"
        a0=$a1; e0=$e1
      done ) > "$RUN/cpu_${M}_rep${rep}.log" 2>&1 &
    CPUMON=$!
    grep -E "Model Runner|Available KV|GPU KV cache size|block size|max_num_batched_tokens|enable_prefix_caching|chunked" "$LOG" | cut -c1-200 | head -8
    # Discarded warm-up with its own seed: bench's --num-warmups replays the first
    # measured request, which would then be a prefix/MM cache hit (seen in the first
    # W1 run, where repeated seeds let later points hit earlier points' caches).
    "$ROOT/envs/vts/bin/vllm" bench serve --backend openai-chat --endpoint /v1/chat/completions \
      --base-url "http://127.0.0.1:$PORT" --model "$M" --tokenizer "$ROOT/models/$M" \
      --dataset-name random-mm --random-input-len 64 --random-output-len 32 --random-range-ratio 0 \
      --random-mm-base-items-per-request 4 --random-mm-num-mm-items-range-ratio 0 \
      --random-mm-limit-mm-per-prompt '{"image": 4, "video": 0}' --random-mm-bucket-config "{($IMG, $IMG, 1): 1.0}" \
      --ignore-eos --temperature 0 --num-prompts 16 --max-concurrency 8 --request-rate inf --seed 999999 \
      > "$RUN/warmup_${M}_rep${rep}.log" 2>&1
    for W in "${WORKLOADS[@]}"; do
      if [ "$W" = P ]; then ITEMS=4; OUT=32; WID=1; else ITEMS=1; OUT=512; WID=2; fi
      for C in "${CONCS[@]}"; do
        N=$(( C * 8 )); [ $N -lt 32 ] && N=32; [ $N -gt 256 ] && N=256
        SEED=$(( rep * 100000 + C * 100 + WID ))   # unique per point, shared by both models (paired)
        TAG=${M}_${W}_c${C}_rep${rep}
        "$ROOT/envs/vts/bin/vllm" bench serve --backend openai-chat --endpoint /v1/chat/completions \
          --base-url "http://127.0.0.1:$PORT" --model "$M" --tokenizer "$ROOT/models/$M" \
          --dataset-name random-mm --random-input-len 64 --random-output-len $OUT --random-range-ratio 0 \
          --random-mm-base-items-per-request $ITEMS --random-mm-num-mm-items-range-ratio 0 \
          --random-mm-limit-mm-per-prompt "{\"image\": $ITEMS, \"video\": 0}" \
          --random-mm-bucket-config "{($IMG, $IMG, 1): 1.0}" \
          --ignore-eos --temperature 0 --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
          --num-prompts $N --max-concurrency $C --request-rate inf --seed $SEED \
          --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
          --save-result --result-dir "$RUN/results" --result-filename "$TAG.json" \
          > "$RUN/results/$TAG.log" 2>&1
        rc=$?
        [ $rc -eq 0 ] || { echo "bench $TAG rc=$rc"; tail -15 "$RUN/results/$TAG.log"; STATUS=1; }
        grep -E "Request throughput|Median TTFT|P99 TTFT|Median TPOT|Output token throughput" "$RUN/results/$TAG.log" | tr -s ' ' | tr '\n' ';' | sed "s/^/$TAG: /"; echo
      done
    done
    echo "cache check rep=$rep $M: $(grep -o "Prefix cache hit rate: [0-9.]*%, MM cache hit rate: [0-9.]*%" "$LOG" | tail -1)"
    smi > "$RUN/nvidia_smi_${M}_rep${rep}.txt"
    kill $SP; wait $SP 2>/dev/null
    kill $DMON $CPUMON 2>/dev/null; wait $DMON $CPUMON 2>/dev/null
    echo "[$(date -Is)] rep=$rep $M stopped"
  done
done
echo "--- nvidia-smi after"; smi | tee "$RUN/nvidia_smi_after.txt"
echo "[$(date -Is)] W1_SERVE_DONE status=$STATUS"
exit $STATUS
