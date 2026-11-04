"""Summarize W4 ③ P / ④ D / ⑤ Pareto serving runs (docs/plans/W4_PLAN.md §5-§8).

Reads <results>/<model>_<W>_<method>_k<keep>_b<budget>_c<conc>_rep<r>.json (vllm bench serve
output) from one or more run dirs, and writes per-point medians / ranges over reps plus the
gains vs B0 (paired per rep, then the median of the per-rep ratios).

Usage: python experiments/w4/serve_summary.py --results <dir> [<dir> ...] --out <json>
"""
import argparse
import json
import os
import re
import statistics as st
from collections import defaultdict

NAME = re.compile(r"^(?P<model>.+?)_(?P<w>[PD])_(?P<method>[a-z0-9_]+?)_k(?P<keep>[0-9.]+)_b(?P<b>\w+)"
                  r"_c(?P<c>\d+)_rep(?P<rep>\d+)\.json$")
METRICS = {
    "req_s": lambda d: d["request_throughput"],
    "prompt_tok_s": lambda d: d["total_input_tokens"] / d["duration"],
    "ttft_p50": lambda d: d["p50_ttft_ms"],
    "ttft_p99": lambda d: d["p99_ttft_ms"],
    "tpot_p50": lambda d: d["p50_tpot_ms"],
}


def load(dirs):
    pts = {}
    for r in dirs:
        for f in sorted(os.listdir(r)):
            m = NAME.match(f)
            if not m:
                continue
            d = json.load(open(os.path.join(r, f), encoding="utf-8"))
            assert d["failed"] == 0 and d["completed"] == d["num_prompts"], f
            g = m.groupdict()
            key = (g["model"], g["w"], g["method"], g["keep"], g["b"], int(g["c"]))
            pts.setdefault(key, {})[int(g["rep"])] = {k: fn(d) for k, fn in METRICS.items()}
    return pts


def summarize(pts):
    rows = []
    for key, reps in sorted(pts.items()):
        model, w, method, keep, b, c = key
        row = dict(model=model, workload=w, method=method, keep=keep, budget=b, conc=c,
                   reps=sorted(reps))
        for k in METRICS:
            v = [reps[r][k] for r in sorted(reps)]
            row[k] = st.median(v)
            row[k + "_range"] = [min(v), max(v)]
        v = [reps[r]["req_s"] for r in reps]
        row["req_s_spread_pct"] = 100 * (max(v) - min(v)) / st.median(v)
        base = pts.get((model, w, "none", "1.0", b, c))
        if base and method != "none":
            common = sorted(set(reps) & set(base))
            row["gain_req_s"] = st.median(reps[r]["req_s"] / base[r]["req_s"] for r in common)
            row["gain_req_s_range"] = [min(reps[r]["req_s"] / base[r]["req_s"] for r in common),
                                       max(reps[r]["req_s"] / base[r]["req_s"] for r in common)]
            row["tpot_change_pct"] = st.median(
                100 * (reps[r]["tpot_p50"] / base[r]["tpot_p50"] - 1) for r in common)
            row["ttft_p50_ratio"] = st.median(reps[r]["ttft_p50"] / base[r]["ttft_p50"] for r in common)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = summarize(load(a.results))
    json.dump(rows, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(f"{len(rows)} points -> {a.out}")


if __name__ == "__main__":
    main()
