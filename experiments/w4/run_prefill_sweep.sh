#!/usr/bin/env bash
# W4 ⑥ prefill length sweep (docs/plans/W4_PLAN.md §9b): 2 models x prefix caching {on, off},
# one process per arm, plugin inactive (VTS_METHOD unset). Does NOT reinstall the plugin.
# SMOKE=1: Qwen3.5 only, lengths 128,822, 1 warm-up, 2 runs.
# PCS (default "on off"), QUANTS (default bf16; fp8l = FP8 on the language model only, follow-up T3),
# LENGTHS (comma list, default: the script's). Output: <run>/<model>/pc_<PC>[_<Q>].json.
# Usage (from the repo root): bash experiments/w4/run_prefill_sweep.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
MODELS=(${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B})
EXTRA=()
[ -n "${LENGTHS:-}" ] && EXTRA=(--lengths "$LENGTHS")
if [ "${SMOKE:-0}" = 1 ]; then MODELS=(Qwen3.5-4B); EXTRA=(--lengths "${LENGTHS:-128,822}" --warmup 1 --runs 2); fi
mkdir -p "$RUN"
unset VTS_METHOD VTS_KEEP
echo "[$(date -Is)] prefill sweep: models=${MODELS[*]} smoke=${SMOKE:-0} gpu=$CUDA_VISIBLE_DEVICES"
STATUS=0
for M in "${MODELS[@]}"; do
  for PC in ${PCS:-on off}; do
   for Q in ${QUANTS:-bf16}; do
    QARG=(); SFX=""
    [ "$Q" = fp8l ] && { QARG=(--quantization fp8_per_tensor --quantization-config '{"ignore": ["visual.*"]}'); SFX=_fp8l; }
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$CUDA_VISIBLE_DEVICES"; then
      echo "[$(date -Is)] GPU busy before $M/pc_$PC, abort"; exit 4
    fi
    timeout 1800 $PY experiments/w4/prefill_sweep.py --model "$ROOT/models/$M" --prefix-caching "$PC" "${EXTRA[@]}" \
      "${QARG[@]}" --out "$RUN/$M/pc_$PC$SFX.json" > "$RUN/${M}_pc_$PC$SFX.log" 2>&1
    rc=$?; [ $rc -ne 0 ] && STATUS=1
    echo "[$(date -Is)] $M/pc_$PC$SFX rc=$rc $(grep -E '^engine ' "$RUN/${M}_pc_$PC$SFX.log" | cut -c1-300)"
    grep -E '^[0-9]+ \{' "$RUN/${M}_pc_$PC$SFX.log" | cut -c1-200
   done
  done
done
echo "[$(date -Is)] SWEEP_DONE status=$STATUS"
exit $STATUS
