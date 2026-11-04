# Shared environment for the experiment runners: `source` it near the top of a run_*.sh.
# Needs VTS_ROOT (holding envs/vts, models/, data/, cache/, tmp/) and CUDA_VISIBLE_DEVICES
# (a GPU UUID; device order is pinned to PCI bus order so indices match nvidia-smi).
ROOT=${VTS_ROOT:?set VTS_ROOT to the directory holding envs/ models/ data/ cache/}
PY=$ROOT/envs/vts/bin/python
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$ROOT/cache/hf TMPDIR=$ROOT/tmp
export VLLM_CACHE_ROOT=$ROOT/cache/vllm TRITON_CACHE_DIR=$ROOT/cache/triton
# FlashInfer's sampler JIT failed on the measurement host; the PyTorch sampler is used throughout.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to a GPU UUID (nvidia-smi -L)}"
