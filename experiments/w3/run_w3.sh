#!/usr/bin/env bash
# W3 accuracy sweep (docs/plans/W3_PLAN.md). Same server settings as W1 accuracy so the
# W1 per-sample results serve as the B0 baseline (paired by doc_id).
# Usage (from the repo root): BATCH=A|B|R|N|T|F|V MODELS="..." bash experiments/w3/run_w3.sh <run_dir>
#   BATCH=R (2026-10-23): original RATE (PR #60963 head 45333ba, method rate_orig) at keep
#   0.5 / 0.25 on the three OCR tasks only, same limits as A/B so it pairs with the same B0.
#   BATCH=N (2026-10-24): run-to-run noise control: B0 (plugin off) and random 50%
#   (no ranking) re-run once each on ChartQA, compared per sample with W1 B0 / batch B.
#   BATCH=T (W5(a)): DivPrune with the Triton kernel at keep 0.5, ChartQA (vs batch B divprune_k50).
#   BATCH=F (W5(b)): FP8 weights (--quantization fp8) x {B0, downsample 0.5, rate 0.5} on the
#   three OCR tasks, same limits as R (vs the bf16 W1 B0 / batch B).
#   BATCH=V (W5c): B0 with FP8 on the language model only (ViT bf16), ChartQA.
#   Method token divprune_triton = divprune with VTS_DIVPRUNE_IMPL=triton.
#   SMOKE_LIMIT=N caps every task at N samples (smoke runs only).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
PORT=${PORT:-8061}
BATCH=${BATCH:?A, B, R, N, T, F or V}
QARG=(); [ "$BATCH" = F ] && QARG=(--quantization fp8)
[ "$BATCH" = V ] && QARG=(--quantization fp8_per_tensor --quantization-config '{"ignore": ["visual.*"]}')
MODELS=(${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B})
mkdir -p "$RUN"

if [ "$BATCH" = A ]; then
  CONFIGS=("random_k25_s0|random 0.25 0" "divprune_k25|divprune 0.25 0" "rate_k25|rate 0.25 0"
           "downsample_k25|downsample 0.25 0" "pool2x2|pool2x2 1.0 0" "rate_pr_k25|rate_pr 0.25 0"
           "random_k25_s1|random 0.25 1" "random_k25_s2|random 0.25 2")
elif [ "$BATCH" = R ]; then
  CONFIGS=("rate_orig_k50|rate_orig 0.5 0" "rate_orig_k25|rate_orig 0.25 0")
elif [ "$BATCH" = T ]; then
  CONFIGS=("divprune_triton_k50|divprune_triton 0.5 0")
elif [ "$BATCH" = F ]; then
  CONFIGS=("b0_fp8|none 1.0 0" "downsample_k50_fp8|downsample 0.5 0" "rate_k50_fp8|rate 0.5 0")
elif [ "$BATCH" = V ]; then
  CONFIGS=("b0_fp8l|none 1.0 0")
elif [ "$BATCH" = N ]; then
  CONFIGS=("b0_rerun|none 1.0 0" "random_k50_rerun|random 0.5 0")
else
  CONFIGS=()
  for k in 50 125; do
    kv=$( [ $k = 50 ] && echo 0.5 || echo 0.125 )
    for m in random divprune rate downsample; do CONFIGS+=("${m}_k${k}|$m $kv 0"); done
  done
fi

$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || exit 5
$PY experiments/w1/make_local_tasks.py --root "$ROOT" > "$RUN/make_local_tasks.log" 2>&1 || exit 5
echo "[$(date -Is)] W3 batch $BATCH: ${#CONFIGS[@]} configs x ${#MODELS[@]} models"

STATUS=0
for M in "${MODELS[@]}"; do
  for C in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME SPEC <<< "$C"
    if [[ " ${SKIP:-} " == *" $M/$NAME "* ]]; then echo "[$(date -Is)] skip $M/$NAME (already valid)"; continue; fi
    read -r METHOD KEEP SEED <<< "$SPEC"
    RENV=()
    [ "$METHOD" = divprune_triton ] && { METHOD=divprune; RENV=(VTS_DIVPRUNE_IMPL=triton); }
    OUT=$RUN/$M/$NAME
    mkdir -p "$OUT"
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
      echo "[$(date -Is)] GPU busy before $M/$NAME, abort"; exit 4
    fi
    if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy"; exit 6; fi
    env VTS_METHOD=$METHOD VTS_KEEP=$KEEP VTS_SEED=$SEED "${RENV[@]}" "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" \
      --served-model-name "$M" --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
      --gpu-memory-utilization 0.90 --max-model-len 32768 --max-num-seqs 64 \
      --limit-mm-per-prompt '{"image": 1, "video": 0}' --seed 0 "${QARG[@]}" > "$OUT/serve.log" 2>&1 &
    SP=$!
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
    grep -c "vts_plugin" "$OUT/serve.log" > "$OUT/plugin_lines.txt"
    t0=$(date +%s)
    MA="model=$M,base_url=http://127.0.0.1:$PORT/v1,api_key=EMPTY,num_concurrent=32,timeout=600,max_size_in_mb=64,enable_thinking_kwarg=False,httpx_trust_env=False"
    if [ "$BATCH" = X ] || [ "$BATCH" = N ] || [ "$BATCH" = T ] || [ "$BATCH" = V ]; then
      GROUPS_=("vts_chartqa|")
    elif [ "$BATCH" = R ] || [ "$BATCH" = F ]; then
      GROUPS_=("vts_chartqa|" "vts_textvqa_val|--limit 1500" "vts_docvqa_val|--limit 1000")
    else
      GROUPS_=("vts_chartqa,vts_mme|" "vts_textvqa_val,vts_mmbench_en_dev_static|--limit 1500" "vts_docvqa_val|--limit 1000")
    fi
    for G in "${GROUPS_[@]}"; do
      IFS='|' read -r TASKS LIM <<< "$G"
      [ -n "${SMOKE_LIMIT:-}" ] && LIM="--limit $SMOKE_LIMIT"
      OPENAI_API_KEY=EMPTY "$ROOT/envs/vts/bin/lmms-eval" eval --model openai --model_args "$MA" \
        --tasks "$TASKS" --include_path "$ROOT/tasks_local" $LIM --batch_size 1 --seed 0 \
        --log_samples --output_path "$OUT/lmms_eval" >> "$OUT/lmms_eval.log" 2>&1 || { echo "lmms-eval $TASKS failed"; STATUS=1; }
    done
    nfail=$(grep -c "All 5 attempts failed" "$OUT/lmms_eval.log")
    noom=$(grep -c -E "OutOfMemoryError|EngineDeadError" "$OUT/serve.log")
    nsplit=$(grep -o -E "split into [0-9]+ calls \([0-9]+ splits so far\)" "$OUT/serve.log" | tail -1)
    valid=yes
    if [ "$nfail" -gt 0 ] || [ "$noom" -gt 0 ]; then valid=NO; STATUS=1; fi
    echo "$valid" > "$OUT/valid.txt"
    echo "[$(date -Is)] $M/$NAME ($METHOD keep=$KEEP seed=$SEED ${RENV[*]}) done in $(( $(date +%s) - t0 ))s, plugin lines $(cat "$OUT/plugin_lines.txt"), failed_requests=$nfail oom_or_dead=$noom valid=$valid vit_split=[${nsplit:-none}]"
    kill $SP; wait $SP 2>/dev/null
  done
done
echo "[$(date -Is)] W3_DONE batch=$BATCH status=$STATUS"
exit $STATUS
