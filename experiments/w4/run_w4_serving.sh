#!/usr/bin/env bash
# W4 ③ P / ④ D / ⑤ Pareto serving runs (docs/plans/W4_PLAN.md §5-§8). Same client and server
# settings as W1 run_serving.sh (random-mm, 896x896, 64 text tokens in, CUDA graphs, vLLM
# defaults for prefix caching), plus the method / keep (plugin env) and, for P, the per-step
# token budget (--max-num-batched-tokens). One server per (rep, budget, method, model).
#   WORKLOAD=P: 4 images, 32 out    WORKLOAD=D: 1 image, 512 out
#   METHODS (default none downsample random rate divprune), KEEP (default 0.5),
#   BUDGETS (P only; default "2048 8192"; D uses the vLLM default), CONCS (default 1 8 32 64)
#   Method tokens (W5, 2026-10-24): divprune_triton = divprune with VTS_DIVPRUNE_IMPL=triton;
#   fp8:<m> = <m> served with --quantization fp8 (online FP8 weights), labelled fp8<m> in the
#   result names, so serve_summary pairs it with the bf16 B0 ("none"). MINN (default 32) is the
#   minimum number of prompts per point. fp8l:<m> (W5c) = FP8 on the language model only:
#   --quantization fp8_per_tensor with quantization_config.ignore = visual.* (ViT stays bf16).
# bi:<m> = <m> with VLLM_BATCH_INVARIANT=1, labelled bi<m>.
# TRACE=1 (follow-ups S2/S4): VTS_VIT_TRACE=<run>/vit/<server> for each server (plugin >= 0.1.6).
# STEPTRACE=1 (follow-up T1): VTS_STEP_TRACE=<run>/steps/<server> (plugin >= 0.1.7).
# METRICS=1: /metrics sampled every 1 s per point into results/<tag>.metrics.csv (as in K1).
# Each point also writes results/<tag>.window (bench start / end, unix time) to cut the trace.
# Seeds are unique per point and shared across models / methods / keeps (paired), disjoint
# from W1 (<= 3e5) and K1 (3e5..3.4e5). Plugin is not reinstalled; its version is checked.
# Usage (from the repo root): WORKLOAD=P bash experiments/w4/run_w4_serving.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
WORKLOAD=${WORKLOAD:?P or D}
MODELS=(${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B})
METHODS=(${METHODS:-none downsample random rate divprune})
KEEP=${KEEP:-0.5}
CONCS=(${CONCS:-1 8 32 64})
REPS=${REPS:-3}
MINN=${MINN:-32}
PORT=${PORT:-8041}
IMG=896
if [ "$WORKLOAD" = P ]; then ITEMS=4; OUT=32; WID=1; BUDGETS=(${BUDGETS:-2048 8192}); else ITEMS=1; OUT=512; WID=2; BUDGETS=(default); fi
mkdir -p "$RUN/results"
ver=$($PY -c "import vts_plugin; print(vts_plugin.__version__)" 2>/dev/null)
want=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)  # the release's own version
[ "$ver" = "$want" ] || { echo "installed vts_plugin is '$ver', expected $want"; exit 5; }

smi() { nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader; }
# grep -c, not grep -q: with pipefail, grep -q exits at the first match, nvidia-smi dies of SIGPIPE and
# the pipeline reports "no match" exactly when another process is there (found 2026-10-28, fixed here 10-29)
gpu_busy() { [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")" -gt 0 ]; }
bench() {  # bench <model> <num_prompts> <concurrency> <seed> [extra...]
  local m=$1 n=$2 c=$3 seed=$4; shift 4
  "$ROOT/envs/vts/bin/vllm" bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "http://127.0.0.1:$PORT" --model "$m" --tokenizer "$ROOT/models/$m" \
    --dataset-name random-mm --random-input-len 64 --random-output-len $OUT --random-range-ratio 0 \
    --random-mm-base-items-per-request $ITEMS --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt "{\"image\": $ITEMS, \"video\": 0}" \
    --random-mm-bucket-config "{($IMG, $IMG, 1): 1.0}" \
    --ignore-eos --temperature 0 --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
    --num-prompts "$n" --max-concurrency "$c" --request-rate inf --seed "$seed" "$@"
}

echo "[$(date -Is)] W4 serving: workload=$WORKLOAD models=${MODELS[*]} methods=${METHODS[*]} keep=$KEEP budgets=${BUDGETS[*]} concs=${CONCS[*]} reps=$REPS gpu=$CUDA_VISIBLE_DEVICES plugin=$ver"
smi | tee "$RUN/nvidia_smi_before.txt"
STATUS=0
for rep in $(seq 1 "$REPS"); do
  for bi in "${!BUDGETS[@]}"; do
    B=${BUDGETS[$bi]}
    for METHOD in "${METHODS[@]}"; do
      for M in "${MODELS[@]}"; do
        if gpu_busy; then echo "[$(date -Is)] GPU busy before rep=$rep b=$B $METHOD $M, abort"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; exit 4; fi
        if [ "$(ss -ltn | grep -c ":$PORT ")" -gt 0 ]; then echo "port $PORT busy"; exit 6; fi
        BASE=${METHOD#*:}; LABEL=${METHOD/:/}
        case $METHOD in
          fp8:*) QARG=(--quantization fp8) ;;
          fp8l:*) QARG=(--quantization fp8_per_tensor --quantization-config '{"ignore": ["visual.*"]}') ;;
          *) QARG=() ;;
        esac
        case $METHOD in
          bi:*) export VLLM_BATCH_INVARIANT=1 ;;
        esac
        case $BASE in
          none) unset VTS_METHOD VTS_KEEP; K=1.0 ;;
          divprune_triton) export VTS_METHOD=divprune VTS_DIVPRUNE_IMPL=triton VTS_KEEP=$KEEP VTS_SEED=0; K=$KEEP ;;
          *) export VTS_METHOD=$BASE VTS_KEEP=$KEEP VTS_SEED=0; K=$KEEP ;;
        esac
        BARG=(); [ "$B" != default ] && BARG=(--max-num-batched-tokens "$B")
        SRV=${M}_${LABEL}_k${K}_b${B}_rep${rep}
        LOG=$RUN/serve_$SRV.log
        if [ "${TRACE:-0}" = 1 ]; then export VTS_VIT_TRACE=$RUN/vit/$SRV; else unset VTS_VIT_TRACE; fi
        if [ "${STEPTRACE:-0}" = 1 ]; then export VTS_STEP_TRACE=$RUN/steps/$SRV; else unset VTS_STEP_TRACE; fi
        "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" --served-model-name "$M" \
          --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --gpu-memory-utilization 0.90 \
          --max-model-len 16384 --limit-mm-per-prompt '{"image": 4, "video": 0}' "${BARG[@]}" "${QARG[@]}" \
          --seed 0 > "$LOG" 2>&1 &
        SP=$!
        unset VTS_METHOD VTS_KEEP VTS_DIVPRUNE_IMPL VTS_VIT_TRACE VTS_STEP_TRACE VLLM_BATCH_INVARIANT
        ok=0
        for _ in $(seq 1 240); do
          curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
          kill -0 $SP 2>/dev/null || break
          sleep 5
        done
        if [ $ok -ne 1 ]; then echo "[$(date -Is)] serve $SRV failed"; grep -E "Error" "$LOG" | tail -5; kill $SP 2>/dev/null; wait $SP 2>/dev/null; STATUS=1; continue; fi
        echo "[$(date -Is)] $SRV healthy; budget_arg=$(grep -o "max_num_batched_tokens': [0-9]*" "$LOG" | head -1 | grep -o '[0-9]*$'); plugin lines $(grep -c vts_plugin "$LOG"); vit cap $(grep -o "ViT patch cap per call = [0-9]*" "$LOG" | head -1 | grep -o "[0-9]*$"); batch invariant lines $(grep -ci "batch.invariant" "$LOG"); quant=$(grep -o "quantization=[a-z0-9_]*" "$LOG" | head -1); $(grep -o "Quantizing [0-9]* layers" "$LOG" | head -1)"
        # DMON_S (follow-up U1): dmon metric groups, e.g. pucv (power/temp, util, clocks, violations);
        # with DMON_S set the log carries the date too (-o DT) so points can be cut by their windows
        if [ -n "${DMON_S:-}" ]; then DMON_ARGS=(-s "$DMON_S" -o DT); else DMON_ARGS=(-s u -o T); fi
        nvidia-smi dmon -i "$CUDA_VISIBLE_DEVICES" "${DMON_ARGS[@]}" -d 1 > "$RUN/dmon_$SRV.log" 2>&1 &
        DMON=$!
        # discarded warm-up with its own seed (no cache sharing with measured points)
        bench "$M" 16 8 999999 > "$RUN/warmup_$SRV.log" 2>&1
        for C in "${CONCS[@]}"; do
          N=$(( C * 8 )); [ $N -lt $MINN ] && N=$MINN; [ $N -gt 256 ] && N=256
          SEED=$(( 400000 + rep * 10000 + bi * 1000 + C * 10 + WID ))
          TAG=${M}_${WORKLOAD}_${LABEL}_k${K}_b${B}_c${C}_rep${rep}
          MS=
          if [ "${METRICS:-0}" = 1 ]; then
            STOP=$RUN/results/$TAG.stop; rm -f "$STOP"
            $PY experiments/w4/metrics_sampler.py --url "http://127.0.0.1:$PORT/metrics" \
              --out "$RUN/results/$TAG.metrics.csv" --stop-file "$STOP" &
            MS=$!
          fi
          echo "$(date +%s.%N) start" > "$RUN/results/$TAG.window"
          bench "$M" "$N" "$C" "$SEED" --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir "$RUN/results" --result-filename "$TAG.json" > "$RUN/results/$TAG.log" 2>&1
          rc=$?
          echo "$(date +%s.%N) end" >> "$RUN/results/$TAG.window"
          [ -n "$MS" ] && { touch "$STOP"; wait $MS 2>/dev/null; }
          [ $rc -eq 0 ] || { echo "bench $TAG rc=$rc"; tail -15 "$RUN/results/$TAG.log"; STATUS=1; }
          echo "$TAG: $(grep -E 'Failed requests|Request throughput|Median TTFT|Median TPOT' "$RUN/results/$TAG.log" | tr -s ' ' | tr '\n' ';')"
        done
        noom=$(grep -c -E 'OutOfMemoryError|EngineDeadError' "$LOG")
        echo "cache check $SRV: $(grep -o 'Prefix cache hit rate: [0-9.]*%, MM cache hit rate: [0-9.]*%' "$LOG" | tail -1); oom_or_dead=$noom"
        [ "$noom" -gt 0 ] && STATUS=1
        kill $SP; wait $SP 2>/dev/null
        kill $DMON 2>/dev/null; wait $DMON 2>/dev/null
      done
    done
  done
done
smi | tee "$RUN/nvidia_smi_after.txt"
echo "[$(date -Is)] W4_SERVE_DONE workload=$WORKLOAD status=$STATUS"
exit $STATUS
