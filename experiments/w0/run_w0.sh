#!/usr/bin/env bash
# W0 acceptance: serve both models, smoke-test images, run a ChartQA subset.
#   1. wait (read-only polling) until the target GPU has >= NEED_MIB free
#   2. vllm serve Qwen3.5-4B (BF16) -> smoke_client (image reaches model + negative control)
#   3. lmms-eval vts_chartqa through the OpenAI-compatible endpoint (--limit LIMIT)
#   4. vllm serve Qwen3-VL-4B-Instruct -> smoke_client only
# Never touches other processes. Writes everything under $RUN.
#
# Usage: bash experiments/w0/run_w0.sh <run_dir>
set -uo pipefail
ROOT=${VTS_ROOT:?set VTS_ROOT to the directory holding envs/ models/ data/ cache/}
RUN=${1:?run dir}
GPU_UUID=${GPU_UUID:?set GPU_UUID (nvidia-smi -L)}
NEED_MIB=${NEED_MIB:-13312}
MEM_UTIL=${MEM_UTIL:-0.26}
LIMIT=${LIMIT:-200}
PORT=${PORT:-8011}
DEADLINE_H=${DEADLINE_H:-12}
PY=$ROOT/envs/vts/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run_w0.log") 2>&1

smi() { nvidia-smi --query-gpu=uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader; \
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; }
free_mib() { nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }

echo "[$(date -Is)] waiting for $GPU_UUID free >= $NEED_MIB MiB (deadline ${DEADLINE_H}h)"
end=$(( $(date +%s) + DEADLINE_H * 3600 ))
while :; do
  f=$(free_mib)
  if [ "$f" -ge "$NEED_MIB" ]; then echo "[$(date -Is)] free=$f MiB, starting"; break; fi
  if [ "$(date +%s)" -ge "$end" ]; then echo "[$(date -Is)] DEADLINE reached (free=$f MiB), giving up"; exit 3; fi
  sleep 300
done
echo "--- nvidia-smi before"; smi | tee "$RUN/nvidia_smi_before.txt"

serve_and_wait() {  # $1 model dir, $2 log name
  if ss -ltn | grep -q ":$PORT "; then echo "port $PORT already in use, refusing (would probe someone else's server)"; SERVER_PID=; return 1; fi
  GPU_UUID=$GPU_UUID MEM_UTIL=$MEM_UTIL PORT=$PORT bash "$HERE/serve.sh" "$1" > "$RUN/$2" 2>&1 &
  SERVER_PID=$!
  echo "[$(date -Is)] vllm serve pid=$SERVER_PID model=$(basename "$1")"
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "[$(date -Is)] server healthy"; return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited early"; tail -40 "$RUN/$2"; return 1; fi
    sleep 5
  done
  echo "server not healthy after 15 min"; return 1
}
stop_server() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; echo "[$(date -Is)] server stopped"; }

export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$ROOT/cache/hf TMPDIR=$ROOT/tmp
STATUS=0

# --- Qwen3.5-4B: serve + smoke + lmms-eval
if serve_and_wait "$ROOT/models/Qwen3.5-4B" serve_qwen35_4b.log; then
  grep -E "KV cache|GPU KV|Maximum concurrency|Available KV|model weights took|Loading weights took" "$RUN/serve_qwen35_4b.log" | tail -8
  $PY "$HERE/smoke_client.py" --base-url "http://127.0.0.1:$PORT" --model Qwen3.5-4B > "$RUN/smoke_qwen35_4b.json"; rc=$?
  cat "$RUN/smoke_qwen35_4b.json"; echo "smoke_qwen35_4b rc=$rc"; [ $rc -eq 0 ] || STATUS=1
  t0=$(date +%s)
  OPENAI_API_KEY=EMPTY $ROOT/envs/vts/bin/lmms-eval eval --model openai \
    --model_args "model=Qwen3.5-4B,base_url=http://127.0.0.1:$PORT/v1,api_key=EMPTY,num_concurrent=4,timeout=120,enable_thinking_kwarg=False,httpx_trust_env=False" \
    --tasks vts_chartqa --include_path "$ROOT/tasks_local" --limit "$LIMIT" --batch_size 1 --seed 0 \
    --log_samples --output_path "$RUN/lmms_eval" > "$RUN/lmms_eval_qwen35_4b.log" 2>&1; rc=$?
  echo "lmms-eval rc=$rc elapsed=$(( $(date +%s) - t0 ))s"; tail -15 "$RUN/lmms_eval_qwen35_4b.log"; [ $rc -eq 0 ] || STATUS=1
  echo "--- nvidia-smi during (Qwen3.5-4B loaded)"; smi | tee "$RUN/nvidia_smi_during.txt"
  stop_server
else
  STATUS=1; stop_server
fi

# --- Qwen3-VL-4B-Instruct: serve + smoke only
if serve_and_wait "$ROOT/models/Qwen3-VL-4B-Instruct" serve_qwen3vl_4b.log; then
  grep -E "KV cache|GPU KV|Maximum concurrency|Available KV|model weights took|Loading weights took" "$RUN/serve_qwen3vl_4b.log" | tail -8
  $PY "$HERE/smoke_client.py" --base-url "http://127.0.0.1:$PORT" --model Qwen3-VL-4B-Instruct > "$RUN/smoke_qwen3vl_4b.json"; rc=$?
  cat "$RUN/smoke_qwen3vl_4b.json"; echo "smoke_qwen3vl_4b rc=$rc"; [ $rc -eq 0 ] || STATUS=1
  stop_server
else
  STATUS=1; stop_server
fi

echo "--- nvidia-smi after"; smi | tee "$RUN/nvidia_smi_after.txt"
echo "[$(date -Is)] W0_DONE status=$STATUS"
exit $STATUS
