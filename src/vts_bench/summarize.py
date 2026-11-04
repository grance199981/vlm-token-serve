"""Summaries: serving gains vs the uncompressed baseline, paired accuracy deltas, K1 steady state.

The statistics are the ones behind the published tables (experiments/w1/summarize_accuracy.py,
experiments/w3/summarize_w3.py, experiments/w4/serve_summary.py, experiments/w4/k1_steady.py);
tests/test_bench_summarize.py checks that they give identical values.
"""
import csv
import glob
import json
import os
import random
import re
import statistics as st
from collections.abc import Sequence

# --- statistics --------------------------------------------------------------------------------

def bootstrap_ci(xs: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean (2000 resamples, seed 0). For paired deltas pass
    the per-sample differences (config - B0 on the same doc_id)."""
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def steady_state(csv_path: str) -> dict:
    """Mean running / waiting / KV usage over the steady part of a /metrics CSV (drop the first
    20% and last 10% of samples) and the preemptions during the point."""
    rows = list(csv.DictReader(open(csv_path)))
    if len(rows) < 10:
        return {"n": len(rows)}
    seg = rows[int(0.2 * len(rows)):int(0.9 * len(rows))]

    def mean(k):
        return sum(float(r[k]) for r in seg) / len(seg)

    return {"n": len(seg), "running": mean("running"), "waiting": mean("waiting"),
            "kv_usage": mean("kv_usage"),
            "preemptions": float(rows[-1]["preemptions"]) - float(rows[0]["preemptions"])}


# --- serving -----------------------------------------------------------------------------------

POINT = re.compile(r"^(?P<model>.+?)_(?P<w>P|D|K1)_(?P<method>[a-z0-9_]+?)_k(?P<keep>[0-9.]+)"
                   r"_b(?P<b>\w+)_c(?P<c>\d+)_rep(?P<rep>\d+)\.json$")
SERVE_METRICS = {
    "req_s": lambda d: d["request_throughput"],
    "ttft_p50": lambda d: d["p50_ttft_ms"],
    "ttft_p99": lambda d: d["p99_ttft_ms"],
    "tpot_p50": lambda d: d["p50_tpot_ms"],
}


def load_serve_points(dirs: Sequence[str]) -> dict:
    """{(model, workload, method, keep, budget, conc): {rep: {metric: value}}}; every point must
    have completed all its requests."""
    pts = {}
    for r in dirs:
        for f in sorted(os.listdir(r)):
            m = POINT.match(f)
            if not m:
                continue
            d = json.load(open(os.path.join(r, f), encoding="utf-8"))
            if d["failed"] != 0 or d["completed"] != d["num_prompts"]:
                raise ValueError(f"{f}: {d['failed']} failed / {d['completed']} of {d['num_prompts']}")
            g = m.groupdict()
            key = (g["model"], g["w"], g["method"], g["keep"], g["b"], int(g["c"]))
            pts.setdefault(key, {})[int(g["rep"])] = {k: fn(d) for k, fn in SERVE_METRICS.items()}
            metrics_csv = os.path.join(r, f[:-5] + ".metrics.csv")
            if os.path.exists(metrics_csv):
                pts[key][int(g["rep"])]["running"] = steady_state(metrics_csv).get("running")
    return pts


def serve_gains(pts: dict) -> list[dict]:
    """Per point: medians over reps, and gains vs the B0 point ("none", keep 1.0) with the same
    model / workload / budget / concurrency, paired per rep, median of the per-rep ratios."""
    rows = []
    for key, reps in sorted(pts.items()):
        model, w, method, keep, b, c = key
        row = dict(model=model, workload=w, method=method, keep=keep, budget=b, conc=c, reps=sorted(reps))
        for k in SERVE_METRICS:
            row[k] = st.median(reps[r][k] for r in reps)
        base = pts.get((model, w, "none", "1.0", b, c))
        if base and method != "none":
            common = sorted(set(reps) & set(base))
            ratios = [reps[r]["req_s"] / base[r]["req_s"] for r in common]
            row["gain_req_s"] = st.median(ratios)
            row["gain_req_s_range"] = [min(ratios), max(ratios)]
            row["tpot_change_pct"] = st.median(100 * (reps[r]["tpot_p50"] / base[r]["tpot_p50"] - 1)
                                               for r in common)
            if all(reps[r].get("running") and base[r].get("running") for r in common):
                row["gain_running"] = st.median(reps[r]["running"] / base[r]["running"] for r in common)
        rows.append(row)
    return rows


# --- accuracy ----------------------------------------------------------------------------------

TASK_METRIC = {"vts_chartqa": "relaxed_overall", "vts_textvqa_val": "exact_match", "vts_docvqa_val": "anls"}


def load_samples(model_dir: str, task: str) -> dict[int, float]:
    """{doc_id: score} from the newest lmms-eval samples file of `task` under model_dir."""
    files = sorted(glob.glob(os.path.join(model_dir, "**", f"*_samples_{task}.jsonl"), recursive=True))
    if not files:
        return {}
    return {int(r["doc_id"]): float(r[TASK_METRIC[task]])
            for r in map(json.loads, open(files[-1], encoding="utf-8"))}


def paired_delta(b0: dict[int, float], cur: dict[int, float]) -> dict:
    """Score of `cur` minus B0 on the shared doc_ids, in points (x100), with a paired bootstrap CI."""
    ids = sorted(set(b0) & set(cur))
    if not ids:
        raise ValueError("no shared doc_ids")
    d = [cur[i] - b0[i] for i in ids]
    lo, hi = bootstrap_ci(d)
    return {"n": len(ids), "b0": 100 * sum(b0[i] for i in ids) / len(ids),
            "score": 100 * sum(cur[i] for i in ids) / len(ids),
            "delta": 100 * sum(d) / len(d), "ci_lo": 100 * lo, "ci_hi": 100 * hi}
