#!/usr/bin/env bash
# Follow-up U1 (docs/plans/FOLLOWUP3_PLAN.md): is the slowdown under load GPU clock throttling?
# B0 bf16 / full FP8, P workload, budget 2048, concurrency 1 / 32 / 96, ViT trace on,
# nvidia-smi dmon -s pucv. No plugin reinstall (run_w4_serving.sh checks the installed version).
# SMOKE=1: Qwen3.5 only, concurrency 1 and 32.
# Usage (from the repo root): bash experiments/w6/run_followups3.sh <run_dir>
set -uo pipefail
RUN=${1:?run dir}
: "${CUDA_VISIBLE_DEVICES:?GPU UUID}"
mkdir -p "$RUN"
if [ "${SMOKE:-0}" = 1 ]; then U1=(MODELS=Qwen3.5-4B CONCS="1 32"); else U1=(MODELS="Qwen3-VL-4B-Instruct Qwen3.5-4B" CONCS="1 32 96"); fi
echo "[$(date -Is)] follow-up U1 smoke=${SMOKE:-0}"
env WORKLOAD=P METHODS="none fp8:none" BUDGETS=2048 REPS=1 TRACE=1 DMON_S=pucv PORT=8095 "${U1[@]}" \
  bash experiments/w4/run_w4_serving.sh "$RUN/u1" > "$RUN/u1.log" 2>&1
rc=$?
echo "[$(date -Is)] phase u1 rc=$rc; $(grep -E 'DONE|abort' "$RUN/u1.log" | tail -1)"
echo "[$(date -Is)] FOLLOWUPS3_DONE status=$rc"
exit $rc
