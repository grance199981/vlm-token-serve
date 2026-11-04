#!/usr/bin/env bash
# Follow-up experiments S1-S4 (docs/plans/FOLLOWUP_PLAN.md), one GPU, phases in order, each in its own
# sub-directory; a failed phase is recorded and the next one still runs.
#   s1a K1 keep 0.25 (896)   s1b K1 keep 0.5 (1344)   s2 FP8 ViT trace (P, conc 64)
#   s3 Qwen3.5 process history (prefix caching off, then on)   s4 B0 P concurrency sweep
# PHASES (default all) picks a subset; SMOKE=1 shrinks the phases (one model / method / point each) to ~30 min.
# Usage (from the repo root): bash experiments/w6/run_followups.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
PHASES=(${PHASES:-s1a s1b s2 s3 s4})
SMOKE=${SMOKE:-0}
mkdir -p "$RUN"
$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || exit 5
echo "[$(date -Is)] follow-ups phases=${PHASES[*]} smoke=$SMOKE plugin=$($PY -c 'import vts_plugin; print(vts_plugin.__version__)')"
MODELS="Qwen3-VL-4B-Instruct Qwen3.5-4B"
if [ "$SMOKE" = 1 ]; then
  K1=(MODELS=Qwen3.5-4B REPS=1 METHODS=random CONCS=32); S2M="fp8l:rate"; S3=(REPS=1 RUNS=3)
  S4=(REPS=1 CONCS=32)
else
  K1=(MODELS="$MODELS" REPS=3 METHODS="none random downsample" CONCS=128)
  S2M="none rate fp8:none fp8:rate fp8l:none fp8l:rate"; S3=(REPS=2 RUNS=20); S4=(REPS=2 CONCS="32 48 64 96")
fi
STATUS=0
phase() {  # phase <name> <command...>
  local name=$1; shift
  [[ " ${PHASES[*]} " == *" $name "* ]] || return 0
  echo "[$(date -Is)] phase $name start"
  "$@" > "$RUN/$name.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || STATUS=1
  echo "[$(date -Is)] phase $name rc=$rc; $(grep -E 'DONE|abort|failed' "$RUN/$name.log" | tail -2 | tr '\n' ' ')"
}
phase s1a env "${K1[@]}" KEEP=0.25 SIDE=896 KV_BYTES=4294967296 PORT=8081 \
  bash experiments/w4/run_k1.sh "$RUN/s1a"
phase s1b env "${K1[@]}" KEEP=0.5 SIDE=1344 KV_BYTES=4294967296 PORT=8082 \
  bash experiments/w4/run_k1.sh "$RUN/s1b"
phase s2 env WORKLOAD=P MODELS="$MODELS" METHODS="$S2M" BUDGETS=2048 CONCS=64 REPS=1 TRACE=1 PORT=8083 \
  bash experiments/w4/run_w4_serving.sh "$RUN/s2"
phase s3 bash -c "env PC=off ${S3[*]} bash experiments/w4/run_proc_rep.sh '$RUN/s3' && env PC=on ${S3[*]} bash experiments/w4/run_proc_rep.sh '$RUN/s3'"
phase s4 env WORKLOAD=P MODELS="$MODELS" METHODS=none BUDGETS=2048 "${S4[@]}" TRACE=1 METRICS=1 PORT=8084 \
  bash experiments/w4/run_w4_serving.sh "$RUN/s4"
echo "[$(date -Is)] FOLLOWUPS_DONE status=$STATUS"
exit $STATUS
