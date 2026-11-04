"""Summarize lmms-eval results from run_accuracy.sh and add bootstrap 95% CIs.

Reads <run_dir>/lmms_eval/<model>/*_results.json for the aggregate metrics and the
matching *_samples_<task>.jsonl for per-sample scores (bootstrap, 2000 resamples,
seed 0). MME is a sum of per-subtask accuracies, so it is reported without a CI.

Usage: python experiments/w1/summarize_accuracy.py <run_dir>
"""
import glob
import json
import os
import random
import sys

PER_SAMPLE = {"vts_chartqa": "relaxed_overall", "vts_textvqa_val": "exact_match",
              "vts_docvqa_val": "anls"}


def bootstrap(xs, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main(run_dir):
    for mdir in sorted(glob.glob(os.path.join(run_dir, "lmms_eval", "*"))):
        model = os.path.basename(mdir)
        res_files = sorted(glob.glob(os.path.join(mdir, "*_results.json")))
        if not res_files:
            continue
        data = json.load(open(res_files[-1], encoding="utf-8"))
        if "results" not in data:  # e.g. lmms_eval/submissions/ holds per-task side files, not a model run
            continue
        res = data["results"]
        print(f"## {model}")
        for task, metrics in sorted(res.items()):
            vals = {k.split(",")[0]: v for k, v in metrics.items()
                    if not k.startswith(("alias", "submission")) and "stderr" not in k and isinstance(v, (int, float))}
            line = f"{task}: " + ", ".join(f"{k}={v:.4g}" for k, v in vals.items())
            key = PER_SAMPLE.get(task)
            samples = sorted(glob.glob(os.path.join(mdir, f"*_samples_{task}.jsonl")))
            if key and samples:
                xs = [float(json.loads(l)[key]) for l in open(samples[-1], encoding="utf-8")]
                lo, hi = bootstrap(xs)
                line += f" | n={len(xs)} mean={sum(xs) / len(xs):.4f} 95%CI=[{lo:.4f}, {hi:.4f}]"
            print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1])
