#!/usr/bin/env bash
# W4 step 1: ① stage micro-benchmark, then the K1 calibration (B0 only), docs/plans/W4_PLAN.md §3, §4.2.
# SMOKE=1: one model, two methods, 2 runs; calibration at concurrency 16.
# Usage (from the repo root): bash experiments/w4/run_step1.sh <run_dir>
set -uo pipefail
RUN=${1:?run dir}
if [ "${SMOKE:-0}" = 1 ]; then
  s1=0
  [ "${SKIP_STAGE:-0}" = 1 ] || { MODELS="Qwen3.5-4B" METHODS="none rate" RUNS=2 bash experiments/w4/run_stage_bench.sh "$RUN/stage"; s1=$?; }
  MODELS="Qwen3.5-4B" CALIBRATE=1 CONCS=16 KV_BYTES=${KV_BYTES:-4294967296} bash experiments/w4/run_k1.sh "$RUN/k1cal"; s2=$?
else
  bash experiments/w4/run_stage_bench.sh "$RUN/stage"; s1=$?
  CALIBRATE=1 KV_BYTES=${KV_BYTES:-4294967296} bash experiments/w4/run_k1.sh "$RUN/k1cal"; s2=$?
fi
echo "[$(date -Is)] STEP1_DONE stage=$s1 k1cal=$s2"
[ $s1 -eq 0 ] && [ $s2 -eq 0 ]
