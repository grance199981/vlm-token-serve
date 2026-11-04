"""Command line: `vts-bench {serve,accuracy,summarize} ...` (see `vts-bench <cmd> -h`)."""
import argparse
import json
import os
import sys
from pathlib import Path

from .accuracy import AccuracyRun, run_accuracy
from .load import LoadRun, run_load
from .presets import WORKLOADS, parse_method
from .summarize import load_samples, load_serve_points, paired_delta, serve_gains


def _common(p: argparse.ArgumentParser):
    p.add_argument("--root", default=os.environ.get("VTS_ROOT"),
                   help="directory with envs/vts, models/, tasks_local/ (default $VTS_ROOT)")
    p.add_argument("--out", required=True, help="run directory (created)")
    p.add_argument("--models", nargs="+", default=["Qwen3-VL-4B-Instruct", "Qwen3.5-4B"])
    p.add_argument("--methods", nargs="+", default=["none", "downsample"],
                   help="none | keep_all | random | pool2x2 | divprune | divprune_triton | rate | "
                        "rate_pr | rate_orig | downsample")
    p.add_argument("--keep", type=float, default=0.5)
    p.add_argument("--port", type=int)
    p.add_argument("--allow-shared-gpu", action="store_true",
                   help="start even if another process holds memory on the GPU (accuracy checks only)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vts-bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="serving benchmark with a workload preset")
    _common(s)
    s.add_argument("--preset", choices=sorted(WORKLOADS), required=True)
    s.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32, 64])
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--budgets", nargs="+", default=None,
                   help="--max-num-batched-tokens values (default: 2048 8192 for P, vLLM's default otherwise)")
    s.add_argument("--kv-bytes", type=int, help="K1: --kv-cache-memory-bytes, same for every model")

    a = sub.add_parser("accuracy", help="lmms-eval accuracy through vllm serve")
    _common(a)
    a.add_argument("--tasks", nargs="+", default=["vts_chartqa"])
    a.add_argument("--limit", type=int, help="cap every task (smoke runs)")

    m = sub.add_parser("summarize", help="gains vs B0 (serve) or paired deltas (accuracy)")
    m.add_argument("kind", choices=["serve", "accuracy"])
    m.add_argument("dirs", nargs="+", help="serve: results/ dirs; accuracy: <b0 model dir> <config dir>")
    m.add_argument("--task", default="vts_chartqa")

    args = ap.parse_args(argv)
    if args.cmd == "summarize":
        if args.kind == "serve":
            print(json.dumps(serve_gains(load_serve_points(args.dirs)), indent=1))
        else:
            if len(args.dirs) != 2:
                ap.error("accuracy summary takes <b0 model dir> <config dir>")
            print(json.dumps(paired_delta(load_samples(args.dirs[0], args.task),
                                          load_samples(args.dirs[1], args.task)), indent=1))
        return 0

    if not args.root:
        ap.error("--root or $VTS_ROOT is required")
    methods = [parse_method(t, args.keep) for t in args.methods]
    gpu = None if args.allow_shared_gpu else os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.cmd == "serve" and args.allow_shared_gpu:
        ap.error("--allow-shared-gpu is for accuracy runs; timing needs a GPU of its own")
    if args.cmd == "serve":
        w = WORKLOADS[args.preset]
        budgets = tuple(args.budgets or (["2048", "8192"] if w.name == "P" else ["default"]))
        run = LoadRun(Path(args.root), Path(args.out), w, args.models, methods, args.concurrency,
                      reps=args.reps, budgets=budgets, kv_bytes=args.kv_bytes,
                      port=args.port or 8041, gpu=gpu)
        return 1 if run_load(run) else 0
    run = AccuracyRun(Path(args.root), Path(args.out), args.models, methods, args.tasks,
                      limit=args.limit, port=args.port or 8061, gpu=gpu)
    return 1 if run_accuracy(run) else 0


if __name__ == "__main__":
    sys.exit(main())
