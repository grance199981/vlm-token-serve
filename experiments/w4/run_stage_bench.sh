#!/usr/bin/env bash
# W4 ① stage micro-benchmark (docs/plans/W4_PLAN.md §3): 2 models x 5 methods at keep 0.5,
# one process per config. W5: method tokens divprune_triton, fp8:<m>, fp8l:<m> as in run_w4_serving.sh.
# Usage (from the repo root): bash experiments/w4/run_stage_bench.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
MODELS=(${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B})
METHODS=(${METHODS:-none downsample random rate divprune})
RUNS=${RUNS:-20}
mkdir -p "$RUN"
$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || exit 5
echo "[$(date -Is)] stage bench: models=${MODELS[*]} methods=${METHODS[*]} runs=$RUNS"
STATUS=0
for M in "${MODELS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
      echo "[$(date -Is)] GPU busy before $M/$METHOD, abort"; exit 4
    fi
    BASE=${METHOD#*:}; LABEL=${METHOD/:/}
    case $METHOD in
      fp8:*) QARG=(--quantization fp8) ;;
      fp8l:*) QARG=(--quantization fp8_per_tensor --quantization-config '{"ignore": ["visual.*"]}') ;;
      *) QARG=() ;;
    esac
    unset VTS_DIVPRUNE_IMPL
    case $BASE in
      none) unset VTS_METHOD VTS_KEEP ;;
      divprune_triton) export VTS_METHOD=divprune VTS_DIVPRUNE_IMPL=triton VTS_KEEP=0.5 VTS_SEED=0 ;;
      *) export VTS_METHOD=$BASE VTS_KEEP=0.5 VTS_SEED=0 ;;
    esac
    timeout 1800 $PY experiments/w4/stage_bench.py --model "$ROOT/models/$M" --runs "$RUNS" "${QARG[@]}" \
      --out "$RUN/$M/$LABEL.json" > "$RUN/${M}_$LABEL.log" 2>&1
    rc=$?; [ $rc -ne 0 ] && STATUS=1
    echo "[$(date -Is)] $M/$LABEL rc=$rc $(grep -E '^(P|D) ' "$RUN/${M}_$LABEL.log" | cut -c1-260 | tr '\n' ' ')"
  done
done
unset VTS_METHOD VTS_KEEP VTS_DIVPRUNE_IMPL
echo "[$(date -Is)] STAGE_DONE status=$STATUS"
exit $STATUS
