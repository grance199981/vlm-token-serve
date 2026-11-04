#!/usr/bin/env bash
# Create the project venv (vllm 0.32.0 + lmms-eval) under $VTS_ROOT/envs/vts.
# Uses the shared anaconda Python 3.12 only as the base interpreter; nothing is
# installed into the shared anaconda. All caches stay under $ROOT because /home
# is 94% full.
#
# Usage: bash experiments/remote/setup_env.sh $VTS_ROOT
set -euo pipefail

ROOT=${1:?usage: setup_env.sh <remote_root>}
BASE_PY=${BASE_PY:?set BASE_PY to a Python 3.12 interpreter}
ENV=$ROOT/envs/vts
VLLM_VERSION=0.32.0
LMMS_EVAL_VERSION=0.7.3

export PIP_CACHE_DIR=$ROOT/cache/pip
export TMPDIR=$ROOT/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$ROOT/logs"

if [ -e "$ENV" ]; then
  echo "refusing to overwrite existing env: $ENV" >&2
  exit 2
fi

echo "[$(date -Is)] base python: $($BASE_PY -V)"
$BASE_PY -m venv "$ENV"
PY=$ENV/bin/python
$PY -m pip install -U pip wheel setuptools

echo "[$(date -Is)] installing vllm==$VLLM_VERSION"
$PY -m pip install "vllm==$VLLM_VERSION"

# Freeze the packages vLLM pins so that installing lmms-eval cannot move them.
$PY -m pip freeze | grep -i -E '^(torch|torchvision|torchaudio|transformers|tokenizers|vllm|flashinfer-python|triton|xformers|numpy|safetensors)==' \
  > "$ROOT/logs/vllm_core_constraints.txt"
echo "core constraints:"; cat "$ROOT/logs/vllm_core_constraints.txt"

echo "[$(date -Is)] installing lmms-eval==$LMMS_EVAL_VERSION"
$PY -m pip install "lmms-eval==$LMMS_EVAL_VERSION" -c "$ROOT/logs/vllm_core_constraints.txt"

echo "[$(date -Is)] pip check"
$PY -m pip check || echo "PIP_CHECK_REPORTED_CONFLICTS"

$PY -m pip freeze > "$ROOT/logs/env_vts_freeze.txt"
$PY - <<'EOF'
import importlib.metadata as md
for p in ["vllm", "torch", "transformers", "flashinfer-python", "lmms_eval", "triton", "numpy"]:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}: NOT INSTALLED")
import torch
print("torch.cuda build:", torch.version.cuda)
EOF
echo "[$(date -Is)] SETUP_ENV_DONE"
