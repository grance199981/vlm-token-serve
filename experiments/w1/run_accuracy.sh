#!/usr/bin/env bash
# W1 accuracy baseline (B0, no compression) for both models on 5 lmms-eval tasks,
# through the same OpenAI-compatible `vllm serve` path the serving benchmark uses.
# Image size = model default (no --mm-processor-kwargs) to compare with official numbers.
# Run from the repo root; set CUDA_VISIBLE_DEVICES to the GPU UUID to use.
#
# Usage: bash experiments/w1/run_accuracy.sh <run_dir> [models...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}; shift
MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || MODELS=(Qwen3-VL-4B-Instruct Qwen3.5-4B)
TASKS=${TASKS:-vts_chartqa,vts_textvqa_val,vts_docvqa_val,vts_mme,vts_mmbench_en_dev_static}
LIMIT=${LIMIT:-}            # empty = full sets
PORT=${PORT:-8021}
MEM_UTIL=${MEM_UTIL:-0.90}
CONC=${CONC:-32}
mkdir -p "$RUN"

smi() { nvidia-smi --query-gpu=uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; }
echo "[$(date -Is)] W1 accuracy: models=${MODELS[*]} tasks=$TASKS limit=${LIMIT:-full} gpu=$CUDA_VISIBLE_DEVICES"
echo "--- nvidia-smi before"; smi | tee "$RUN/nvidia_smi_before.txt"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
  echo "target GPU is not empty, refusing"; exit 4
fi
$PY experiments/w1/make_local_tasks.py --root "$ROOT" > "$RUN/make_local_tasks.log" 2>&1 || { cat "$RUN/make_local_tasks.log"; exit 5; }

STATUS=0
for M in "${MODELS[@]}"; do
  if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy"; exit 6; fi
  "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" --served-model-name "$M" \
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len 32768 --max-num-seqs 64 --limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --seed 0 > "$RUN/serve_$M.log" 2>&1 &
  SP=$!
  echo "[$(date -Is)] serve $M pid=$SP"
  ok=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
    kill -0 $SP 2>/dev/null || break
    sleep 5
  done
  if [ $ok -ne 1 ]; then echo "serve $M failed"; tail -30 "$RUN/serve_$M.log"; kill $SP 2>/dev/null; STATUS=1; continue; fi
  grep -E "Model Runner|Available KV|GPU KV cache size|block size" "$RUN/serve_$M.log" | cut -c1-220
  t0=$(date +%s)
  OPENAI_API_KEY=EMPTY "$ROOT/envs/vts/bin/lmms-eval" eval --model openai \
    --model_args "model=$M,base_url=http://127.0.0.1:$PORT/v1,api_key=EMPTY,num_concurrent=$CONC,timeout=600,max_size_in_mb=64,enable_thinking_kwarg=False,httpx_trust_env=False" \
    --tasks "$TASKS" --include_path "$ROOT/tasks_local" ${LIMIT:+--limit $LIMIT} --batch_size 1 --seed 0 \
    --log_samples --output_path "$RUN/lmms_eval" > "$RUN/lmms_eval_$M.log" 2>&1
  rc=$?
  echo "[$(date -Is)] lmms-eval $M rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
  grep -E "^\|" "$RUN/lmms_eval_$M.log" | head -30
  [ $rc -eq 0 ] || STATUS=1
  kill $SP; wait $SP 2>/dev/null
  echo "[$(date -Is)] serve $M stopped"
done
echo "--- nvidia-smi after"; smi | tee "$RUN/nvidia_smi_after.txt"
echo "[$(date -Is)] W1_ACC_DONE status=$STATUS"
exit $STATUS
