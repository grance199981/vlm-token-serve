#!/usr/bin/env bash
# W5 (docs/plans/W5_PLAN.md): (a) Triton DivPrune, (b) FP8 weights x token compression, one GPU,
# phases in order, each in its own sub-directory; a failed phase is recorded and the next
# one still runs (every sub-script checks that the GPU is free before each server).
#   a1 GPU tests  a2 operator bench  a3 stage bench  a4 P serving  a5 ChartQA (batch T)
#   b1 stage bench fp8  b2 P serving fp8  b3 D serving fp8  b4 OCR accuracy fp8 (batch F)
# PHASES (default all) picks a subset; SMOKE=1 shrinks the phases (fewer methods, 1 rep,
# concurrency 8, 10 samples per task; b3 skipped, same code path as b2) to about 30 minutes.
# Usage (from the repo root): bash experiments/w5/run_w5.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
PHASES=(${PHASES:-a1 a2 a3 a4 a5 b1 b2 b3 b4})
SMOKE=${SMOKE:-0}
mkdir -p "$RUN"
$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || exit 5
echo "[$(date -Is)] W5 phases=${PHASES[*]} smoke=$SMOKE plugin=$($PY -c 'import vts_plugin; print(vts_plugin.__version__)')"
if [ "$SMOKE" = 1 ]; then
  BREPS=3; SRUNS=2; SV=(REPS=1 CONCS=8); W3=(SMOKE_LIMIT=10)
  M_A3="none divprune_triton"; M_A4="divprune_triton"; M_B1="fp8:none"; M_B="fp8:rate"
  PHASES=(${PHASES[@]/b3/})
else
  BREPS=30; SRUNS=20; SV=(REPS=3 CONCS="1 64"); W3=()
  M_A3="none divprune divprune_triton"; M_A4="none random divprune divprune_triton"
  M_B1="none fp8:none"; M_B="none downsample rate fp8:none fp8:downsample fp8:rate"
fi
MODELS="Qwen3-VL-4B-Instruct Qwen3.5-4B"
STATUS=0
LAST_RC=0
phase() {  # phase <name> <command...>
  local name=$1; shift
  [[ " ${PHASES[*]} " == *" $name "* ]] || return 0
  echo "[$(date -Is)] phase $name start"
  "$@" > "$RUN/$name.log" 2>&1
  local rc=$?
  LAST_RC=$rc
  [ $rc -eq 0 ] || STATUS=1
  echo "[$(date -Is)] phase $name rc=$rc; $(grep -E 'DONE|passed|failed|INVALID|abort' "$RUN/$name.log" | tail -2 | tr '\n' ' ')"
}
phase a1 env PYTHONPATH=src $PY -m pytest -q -p no:cacheprovider tests/test_vts_divprune_triton.py tests/test_vts_select.py
# a failed correctness test stops the phases that use the kernel (a4, a5); b still runs
if [[ " ${PHASES[*]} " == *" a1 "* ]] && [ "$LAST_RC" -ne 0 ]; then
  echo "[$(date -Is)] a1 failed: skipping a4 a5"; PHASES=(${PHASES[@]/a4/}); PHASES=(${PHASES[@]/a5/})
fi
phase a2 $PY experiments/w5/divprune_bench.py --out "$RUN/a2_divprune_bench.json" --reps $BREPS
phase a3 env MODELS="$MODELS" METHODS="$M_A3" RUNS=$SRUNS bash experiments/w4/run_stage_bench.sh "$RUN/a3"
phase a4 env WORKLOAD=P MODELS="$MODELS" METHODS="$M_A4" BUDGETS=2048 \
  PORT=8051 "${SV[@]}" bash experiments/w4/run_w4_serving.sh "$RUN/a4"
phase a5 env BATCH=T MODELS="$MODELS" PORT=8052 "${W3[@]}" bash experiments/w3/run_w3.sh "$RUN/a5"
phase b1 env MODELS="$MODELS" METHODS="$M_B1" RUNS=$SRUNS bash experiments/w4/run_stage_bench.sh "$RUN/b1"
phase b2 env WORKLOAD=P MODELS="$MODELS" METHODS="$M_B" \
  BUDGETS=2048 PORT=8053 "${SV[@]}" bash experiments/w4/run_w4_serving.sh "$RUN/b2"
phase b3 env WORKLOAD=D MODELS="$MODELS" METHODS="$M_B" \
  MINN=16 PORT=8054 "${SV[@]}" bash experiments/w4/run_w4_serving.sh "$RUN/b3"
phase b4 env BATCH=F MODELS="$MODELS" PORT=8055 "${W3[@]}" bash experiments/w3/run_w3.sh "$RUN/b4"
echo "[$(date -Is)] W5_DONE status=$STATUS"
exit $STATUS
