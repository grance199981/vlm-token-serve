#!/usr/bin/env bash
# Same requests through two builds of the plugin: this checkout (PYTHONPATH=src) and the plugin
# installed in envs/vts. Requests go one at a time (fixed batching), so the two builds should
# give the same answers unless the code changed behaviour. Nothing is installed or reinstalled.
# Usage (from the repo root): bash experiments/w2/compare_builds.sh <run_dir>
#   MODELS (default both), METHODS (default "random rate divprune_triton"), N_CHART (default 20)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"  # ROOT, PY, caches, GPU check
RUN=${1:?run dir}
MODELS=(${MODELS:-Qwen3-VL-4B-Instruct Qwen3.5-4B})
METHODS=(${METHODS:-random rate divprune_triton})
N_CHART=${N_CHART:-20}
PORT=${PORT:-8091}
KEEP=${KEEP:-0.5}
HERE=$(pwd)
mkdir -p "$RUN"
echo "[$(date -Is)] compare builds: models=${MODELS[*]} methods=${METHODS[*]} n=$N_CHART gpu=$CUDA_VISIBLE_DEVICES"
STATUS=0
for M in "${MODELS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    for BUILD in checkout installed; do
      if [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")" -gt 0 ]; then
        echo "[$(date -Is)] GPU busy before $M $METHOD $BUILD, abort"; exit 4
      fi
      TAG=${M}_${METHOD}_${BUILD}
      ENVS=(VTS_KEEP=$KEEP VTS_SEED=0)
      case $METHOD in
        divprune_triton) ENVS+=(VTS_METHOD=divprune VTS_DIVPRUNE_IMPL=triton) ;;
        *) ENVS+=(VTS_METHOD=$METHOD) ;;
      esac
      [ "$BUILD" = checkout ] && ENVS+=(PYTHONPATH=$HERE/src)
      env "${ENVS[@]}" "$ROOT/envs/vts/bin/vllm" serve "$ROOT/models/$M" --served-model-name "$M" \
        --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --gpu-memory-utilization 0.85 \
        --max-model-len 16384 --limit-mm-per-prompt '{"image": 2, "video": 0}' --seed 0 > "$RUN/serve_$TAG.log" 2>&1 &
      SP=$!
      ok=0
      for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { ok=1; break; }
        kill -0 $SP 2>/dev/null || break
        sleep 5
      done
      if [ $ok -ne 1 ]; then echo "[$(date -Is)] $TAG: server failed"; tail -5 "$RUN/serve_$TAG.log"; kill $SP 2>/dev/null; wait $SP 2>/dev/null; STATUS=1; continue; fi
      $PY experiments/w2/integration_client.py --base-url "http://127.0.0.1:$PORT" --model "$M" \
        --model-dir "$ROOT/models/$M" --out "$RUN/$TAG.json" --n-chart "$N_CHART" --n-text 0 \
        --single-only --passes 1 > "$RUN/client_$TAG.log" 2>&1 || STATUS=1
      echo "[$(date -Is)] $TAG: $(grep -o 'vts_plugin [0-9.]* registered' "$RUN/serve_$TAG.log" | head -1); plugin lines $(grep -c vts_plugin "$RUN/serve_$TAG.log"); vit cap $(grep -o 'ViT patch cap per call = [0-9]*' "$RUN/serve_$TAG.log" | head -1 | grep -o '[0-9]*$'); oom $(grep -c -E 'OutOfMemoryError|EngineDeadError' "$RUN/serve_$TAG.log")"
      kill $SP; wait $SP 2>/dev/null
    done
  done
done
$PY - "$RUN" <<'EOF' | tee "$RUN/compare.txt"
import glob, json, os, sys
run = sys.argv[1]
total = same = 0
for f in sorted(glob.glob(os.path.join(run, "*_checkout.json"))):
    g = f.replace("_checkout.json", "_installed.json")
    if not os.path.exists(g):
        print(os.path.basename(f), "no installed-build result"); continue
    a, b = json.load(open(f)), json.load(open(g))
    n = sum(x["output"] == y["output"] and x["prompt_tokens"] == y["prompt_tokens"] for x, y in zip(a, b))
    total += len(a); same += n
    print(f"{os.path.basename(f)[:-14]}: identical {n}/{len(a)}, prompt tokens {a[0]['prompt_tokens']}")
print(f"ALL identical {same}/{total}")
EOF
echo "[$(date -Is)] COMPARE_BUILDS_DONE status=$STATUS"
exit $STATUS
