#!/usr/bin/env bash
# Start `vllm serve` for the W0 smoke test on a shared GPU.
# Not a timing setup: enforce-eager, small batch and a low memory share so the
# user's own jobs on the same card keep running.
#
# Usage: GPU_UUID=GPU-... MEM_UTIL=0.24 PORT=8000 bash experiments/w0/serve.sh <model_dir> [extra vllm args...]
set -euo pipefail
ROOT=${VTS_ROOT:?set VTS_ROOT to the directory holding envs/ models/ data/ cache/}
MODEL=${1:?model dir}; shift
: "${GPU_UUID:?set GPU_UUID}"
MEM_UTIL=${MEM_UTIL:-0.24}
PORT=${PORT:-8000}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_UUID
export HF_HUB_OFFLINE=1
export HF_HOME=$ROOT/cache/hf
export VLLM_CACHE_ROOT=$ROOT/cache/vllm
export TRITON_CACHE_DIR=$ROOT/cache/triton
export TMPDIR=$ROOT/tmp
# FlashInfer's sampler JIT failed on this host with an earlier vLLM;
# keep the same workaround until it is re-tested on 0.32.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

exec "$ROOT/envs/vts/bin/vllm" serve "$MODEL" \
  --served-model-name "$(basename "$MODEL")" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$MEM_UTIL" \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"size": {"longest_edge": 1003520, "shortest_edge": 65536}}' \
  --enforce-eager \
  --seed 0 \
  "$@"
