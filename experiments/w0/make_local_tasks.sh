#!/usr/bin/env bash
# Generate lmms-eval task variants that read our verified local copies instead of
# the hub (the server cannot reach huggingface.co). The local repos are byte-identical
# to lmms-lab-encoder/* (same commit sha and LFS hashes, checked 2026-10-19).
# Only dataset_path / task name change; prompts, splits and metrics are untouched.
#
# Usage: bash experiments/w0/make_local_tasks.sh $VTS_ROOT
set -euo pipefail
ROOT=${1:?remote root}
PY=$ROOT/envs/vts/bin/python
SRC=$($PY -c "import lmms_eval, os; print(os.path.join(os.path.dirname(lmms_eval.__file__), 'tasks'))")
OUT=$ROOT/tasks_local
mkdir -p "$OUT"

# chartqa: single yaml + utils.py
mkdir -p "$OUT/vts_chartqa"
cp "$SRC/chartqa/utils.py" "$OUT/vts_chartqa/utils.py"
sed -e "s#^dataset_path: lmms-lab-encoder/ChartQA#dataset_path: $ROOT/data/lmms-lab/ChartQA#" \
    -e 's#^task: "chartqa"#task: "vts_chartqa"#' \
    -e '/^dataset_kwargs:/,/^  token: True/d' \
    "$SRC/chartqa/chartqa.yaml" > "$OUT/vts_chartqa/vts_chartqa.yaml"

grep -q "dataset_path: $ROOT/data/lmms-lab/ChartQA" "$OUT/vts_chartqa/vts_chartqa.yaml"
grep -q 'task: "vts_chartqa"' "$OUT/vts_chartqa/vts_chartqa.yaml"
! grep -q "lmms-lab-encoder" "$OUT/vts_chartqa/vts_chartqa.yaml"
echo "generated: $OUT/vts_chartqa"
diff "$SRC/chartqa/chartqa.yaml" "$OUT/vts_chartqa/vts_chartqa.yaml" || true
