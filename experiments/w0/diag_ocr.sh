#!/usr/bin/env bash
# W0 diagnosis: both models misread the digit image. Separate causes:
#   A  vLLM, no --mm-processor-kwargs
#   B  vLLM, current serve.sh kwargs (size longest_edge 1003520)
#   C  HF transformers reference (no vLLM)
# Usage: MODEL=Qwen3-VL-4B-Instruct bash experiments/w0/diag_ocr.sh <run_dir>
set -uo pipefail
ROOT=${VTS_ROOT:?set VTS_ROOT to the directory holding envs/ models/ data/ cache/}
RUN=${1:?run dir}; MODEL=${MODEL:-Qwen3-VL-4B-Instruct}
GPU_UUID=${GPU_UUID:?set GPU_UUID (nvidia-smi -L)}
NEED_MIB=${NEED_MIB:-13312}; PORT=${PORT:-8011}
PY=$ROOT/envs/vts/bin/python; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$RUN"; exec > >(tee -a "$RUN/diag_${MODEL}.log") 2>&1
export NO_PROXY=127.0.0.1,localhost HF_HUB_OFFLINE=1 HF_HOME=$ROOT/cache/hf TMPDIR=$ROOT/tmp
f=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[ "$f" -ge "$NEED_MIB" ] || { echo "free=$f MiB < $NEED_MIB, abort"; exit 3; }
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

run_vllm() {  # $1 label, rest: extra serve args
  label=$1; shift
  if ss -ltn | grep -q ":$PORT "; then echo "port busy"; return 1; fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_UUID VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_CACHE_ROOT=$ROOT/cache/vllm \
    "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$MODEL" --served-model-name "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --gpu-memory-utilization 0.26 \
    --max-model-len 8192 --max-num-seqs 4 --limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --enforce-eager --seed 0 "$@" > "$RUN/diag_serve_${MODEL}_${label}.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 $pid 2>/dev/null || { tail -30 "$RUN/diag_serve_${MODEL}_${label}.log"; return 1; }; sleep 5; done
  grep -E "model runner|Model Runner" "$RUN/diag_serve_${MODEL}_${label}.log" | head -3
  for s in 0 1 2 3 4; do
    $PY "$HERE/smoke_client.py" --base-url "http://127.0.0.1:$PORT" --model "$MODEL" --seed $s \
      | $PY -c "import sys,json; d=json.load(sys.stdin); print('$label', 'seed=$s', d['number'], repr(d['answer']), d['prompt_tokens_with_image'])"
  done
  kill $pid; wait $pid 2>/dev/null
}

echo "== A: vLLM, no mm-processor-kwargs"; run_vllm A
echo "== B: vLLM, size longest_edge 1003520"; run_vllm B --mm-processor-kwargs '{"size": {"longest_edge": 1003520, "shortest_edge": 65536}}'
echo "== C: HF transformers reference"
cd "$HERE" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_UUID $PY hf_reference.py --model "$ROOT/models/$MODEL"
echo "DIAG_DONE"
