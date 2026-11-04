#!/usr/bin/env bash
# W4 ⑥c (W4_RESULTS §4.2): is the Qwen3.5 D-shape prefill time a per-process effect or a P->D order
# effect? Qwen3.5 B0, prefix caching default (on), 3 processes "D only" and 3 processes "P then D",
# interleaved (D, PD, D, PD, ...); TRITON_PRINT_AUTOTUNING=1 logs the config each autotuned kernel picks.
# PC=default|on|off (follow-up S3, 2026-10-24): prefix caching passed to stage_bench.py; tags get a _pc<PC> suffix.
# MODES (default "D PD"); RESET=1 (follow-up T4) adds --reset-prefix-cache and a _reset tag suffix.
# Plugin inactive, no reinstall. Usage (from the repo root): bash experiments/w4/run_proc_rep.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
M=${MODEL:-Qwen3.5-4B}
REPS=${REPS:-3}
mkdir -p "$RUN/$M"
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_DEVICE_ORDER=PCI_BUS_ID TRITON_PRINT_AUTOTUNING=1
unset VTS_METHOD VTS_KEEP
echo "[$(date -Is)] proc rep: model=$M reps=$REPS gpu=$CUDA_VISIBLE_DEVICES TRITON_CACHE_AUTOTUNING=${TRITON_CACHE_AUTOTUNING:-unset}"
STATUS=0
for r in $(seq 1 "$REPS"); do
  for MODE in ${MODES:-D PD}; do
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
      echo "[$(date -Is)] GPU busy before $MODE rep $r, abort"; exit 4
    fi
    SH=D; [ "$MODE" = PD ] && SH=P,D
    RARG=(); RSFX=""; [ "${RESET:-0}" = 1 ] && { RARG=(--reset-prefix-cache); RSFX=_reset; }
    TAG=${MODE}_r${r}_pc${PC:-default}$RSFX
    timeout 1800 $PY experiments/w4/stage_bench.py --model "$ROOT/models/$M" --shapes "$SH" --runs "${RUNS:-20}" \
      --prefix-caching "${PC:-default}" "${RARG[@]}" \
      --out "$RUN/$M/$TAG.json" > "$RUN/${M}_$TAG.log" 2>&1
    rc=$?; [ $rc -ne 0 ] && STATUS=1
    echo "[$(date -Is)] $TAG rc=$rc autotune_lines=$(grep -c 'best config selected' "$RUN/${M}_$TAG.log") $(grep -E '^D ' "$RUN/${M}_$TAG.log" | grep -oE '"(ttft_ms|llm_prefill_ms|llm_calls|ttft_iqr_ms)": [0-9.]+' | tr '\n' ' ')"
  done
done
echo "[$(date -Is)] PROCREP_DONE status=$STATUS"
exit $STATUS
