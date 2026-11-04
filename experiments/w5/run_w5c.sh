#!/usr/bin/env bash
# W5c (docs/plans/W5_PLAN.md §5): FP8 on the language model only (ViT kept bf16) vs full online FP8,
# to test whether the sub-multiplicative pruning + FP8 throughput comes from the ViT.
#   c1 stage bench (none, fp8:none, fp8l:none)   c2 P serving, budget 2048, conc 1 / 64, 3 reps
#   c3 ChartQA accuracy, B0 with LLM-only FP8 (batch V)
# PHASES (default all) picks a subset; SMOKE=1 shrinks the phases to about 15 minutes.
# Usage (from the repo root): bash experiments/w5/run_w5c.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
PHASES=(${PHASES:-c1 c2 c3})
SMOKE=${SMOKE:-0}
mkdir -p "$RUN"
$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || exit 5
echo "[$(date -Is)] W5c phases=${PHASES[*]} smoke=$SMOKE plugin=$($PY -c 'import vts_plugin; print(vts_plugin.__version__)')"
if [ "$SMOKE" = 1 ]; then
  SRUNS=2; SV=(REPS=1 CONCS=8); W3=(SMOKE_LIMIT=10)
  M_C1="fp8l:none"; M_C2="fp8l:rate"
else
  SRUNS=20; SV=(REPS=3 CONCS="1 64"); W3=()
  M_C1="none fp8:none fp8l:none"
  M_C2="none downsample rate fp8:none fp8:rate fp8l:none fp8l:downsample fp8l:rate"
fi
MODELS="Qwen3-VL-4B-Instruct Qwen3.5-4B"
STATUS=0
phase() {  # phase <name> <command...>
  local name=$1; shift
  [[ " ${PHASES[*]} " == *" $name "* ]] || return 0
  echo "[$(date -Is)] phase $name start"
  "$@" > "$RUN/$name.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || STATUS=1
  echo "[$(date -Is)] phase $name rc=$rc; $(grep -E 'DONE|INVALID|abort' "$RUN/$name.log" | tail -2 | tr '\n' ' ')"
}
phase c1 env MODELS="$MODELS" METHODS="$M_C1" RUNS=$SRUNS bash experiments/w4/run_stage_bench.sh "$RUN/c1"
phase c2 env WORKLOAD=P MODELS="$MODELS" METHODS="$M_C2" BUDGETS=2048 \
  PORT=8056 "${SV[@]}" bash experiments/w4/run_w4_serving.sh "$RUN/c2"
phase c3 env BATCH=V MODELS="$MODELS" PORT=8057 "${W3[@]}" bash experiments/w3/run_w3.sh "$RUN/c3"
echo "[$(date -Is)] W5C_DONE status=$STATUS"
exit $STATUS
