#!/usr/bin/env bash
# Second follow-ups T1-T5 (docs/plans/FOLLOWUP2_PLAN.md), one GPU, phases in order, each in its own
# sub-directory; a failed phase is recorded and the next one still runs.
#   t1 B0 P concurrency sweep with the engine step trace   t2 FP8 vs bf16 ViT at concurrency 1 (P, D)
#   t3 text-only prefill, bf16 vs LLM-only FP8   t4 Qwen3.5 process history with a prefix-cache reset
#   t5 Qwen3.5 K1 B0 with output 456 vs 480 tokens
# PHASES (default all) picks a subset; SMOKE=1 shrinks every phase (~20 min).
# Usage (from the repo root): bash experiments/w6/run_followups2.sh <run_dir>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
PHASES=(${PHASES:-t1 t2 t3 t4 t5})
SMOKE=${SMOKE:-0}
mkdir -p "$RUN"
$PY -m pip install --no-deps --force-reinstall --no-build-isolation . > "$RUN/pip_install.log" 2>&1 || exit 5
echo "[$(date -Is)] follow-ups 2 phases=${PHASES[*]} smoke=$SMOKE plugin=$($PY -c 'import vts_plugin; print(vts_plugin.__version__)')"
MODELS="Qwen3-VL-4B-Instruct Qwen3.5-4B"
if [ "$SMOKE" = 1 ]; then
  T1=(MODELS=Qwen3.5-4B CONCS=32 REPS=1); T2M=Qwen3.5-4B; T2D=0; T3=(SMOKE=1)
  T4=(REPS=1 RUNS=3); T5=(REPS=1 CONCS=32)
else
  T1=(MODELS="$MODELS" CONCS="32 64 96" REPS=2); T2M=$MODELS; T2D=1; T3=(MODELS="$MODELS")
  T4=(REPS=2 RUNS=20); T5=(REPS=3 CONCS=128)
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
phase t1 env WORKLOAD=P METHODS=none BUDGETS=2048 "${T1[@]}" STEPTRACE=1 METRICS=1 PORT=8091 \
  bash experiments/w4/run_w4_serving.sh "$RUN/t1"
phase t2 bash -c "env WORKLOAD=P MODELS='$T2M' METHODS='none fp8:none' BUDGETS=2048 CONCS=1 REPS=1 TRACE=1 PORT=8092 \
  bash experiments/w4/run_w4_serving.sh '$RUN/t2p' && { [ $T2D = 0 ] || env WORKLOAD=D MODELS='$T2M' METHODS='none fp8:none' \
  CONCS=1 REPS=1 MINN=16 TRACE=1 PORT=8093 bash experiments/w4/run_w4_serving.sh '$RUN/t2d'; }"
phase t3 env "${T3[@]}" PCS=on QUANTS="bf16 fp8l" LENGTHS=1612,3180 bash experiments/w4/run_prefill_sweep.sh "$RUN/t3"
phase t4 bash -c "env PC=on ${T4[*]} bash experiments/w4/run_proc_rep.sh '$RUN/t4' && env PC=on RESET=1 MODES=PD ${T4[*]} bash experiments/w4/run_proc_rep.sh '$RUN/t4'"
phase t5 bash -c "env MODELS=Qwen3.5-4B METHODS=none ${T5[*]} KEEP=0.5 SIDE=896 OUT=456 KV_BYTES=4294967296 PORT=8094 \
  bash experiments/w4/run_k1.sh '$RUN/t5_o456' && env MODELS=Qwen3.5-4B METHODS=none ${T5[*]} KEEP=0.5 SIDE=896 OUT=480 \
  KV_BYTES=4294967296 PORT=8094 bash experiments/w4/run_k1.sh '$RUN/t5_o480'"
echo "[$(date -Is)] FOLLOWUPS2_DONE status=$STATUS"
exit $STATUS
