"""K1 summary against the hypotheses locked in docs/plans/W4_PLAN.md §4.3-4.4.

Gain = steady running concurrency (k1_steady.py) of a method / B0, paired per rep and concurrency;
reported as the mean over reps (with min-max). Points listed in --replace-dir (same file names)
override the main run (used for the 4 points contaminated on 2026-10-22).

Usage: python k1_summary.py --results <run>/results [--replace-dir <redo run>/results] --out <json>
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1_steady import steady  # noqa: E402

MODELS = {"Qwen3-VL-4B-Instruct": "dense", "Qwen3.5-4B": "hybrid"}
METHODS = ["none", "downsample", "random", "rate", "divprune"]
PRUNE = ["random", "rate", "divprune"]
CONCS = [32, 64, 128, 256]
REPS = [1, 2, 3]
PRED = {"dense": {"prune": 1.79, "downsample": 1.76}, "hybrid": {"prune": 1.33, "downsample": 1.32}}


def point(dirs, tag):
    for d in dirs:  # first directory that has the point wins
        if os.path.exists(os.path.join(d, tag + ".json")):
            b = json.load(open(os.path.join(d, tag + ".json")))
            s = steady(os.path.join(d, tag + ".metrics.csv"))
            return {"src": d, "running": s["running"], "kv_usage": s["kv_usage"], "waiting": s["waiting"],
                    "preemptions": s["preemptions"], "n_steady": s["n"], "failed": b["failed"],
                    "req_s": b["request_throughput"], "out_tok_s": b["output_throughput"],
                    "ttft_p50": b["median_ttft_ms"], "tpot_p50": b["median_tpot_ms"],
                    "prompt_tokens": b["total_input_tokens"] / b["completed"]}
    raise SystemExit(f"missing point {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--replace-dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirs = [a.replace_dir, a.results] if a.replace_dir else [a.results]
    pts = {(m, me, c, r): point(dirs, f"{m}_{me}_c{c}_rep{r}")
           for m in MODELS for me in METHODS for c in CONCS for r in REPS}
    replaced = sorted("_".join(map(str, k)) for k, v in pts.items() if a.replace_dir and v["src"] == a.replace_dir)
    assert all(v["failed"] == 0 for v in pts.values()), "failed requests"

    table = {}
    for m in MODELS:
        for me in METHODS:
            for c in CONCS:
                row = {}
                for k in ("running", "req_s", "out_tok_s", "kv_usage", "preemptions", "ttft_p50", "tpot_p50", "prompt_tokens"):
                    row[k] = statistics.mean(pts[(m, me, c, r)][k] for r in REPS)
                for k in ("running", "req_s"):
                    g = [pts[(m, me, c, r)][k] / pts[(m, "none", c, r)][k] for r in REPS]
                    row[k + "_gain"] = (statistics.mean(g), min(g), max(g))
                table[f"{m}|{me}|{c}"] = row

    def gain(m, me, c, k="running"):
        return table[f"{m}|{me}|{c}"][k + "_gain"][0]

    hyp = {}
    for c in CONCS:
        h = {}
        for m, arch in MODELS.items():
            pr = statistics.median(gain(m, me, c) for me in PRUNE)
            ds = gain(m, "downsample", c)
            prs = [gain(m, me, c) for me in PRUNE]
            h[arch] = {"prune_median": pr, "downsample": ds,
                       "K3_prune_ratio": pr / PRED[arch]["prune"], "K3_down_ratio": ds / PRED[arch]["downsample"],
                       "K1b_down_over_prune": ds / pr, "K4_spread": (max(prs) - min(prs)) / min(prs)}
        h["K1_rel_gap"] = (h["dense"]["prune_median"] - h["hybrid"]["prune_median"]) / h["dense"]["prune_median"]
        hyp[c] = h
    p = hyp[128]  # primary concurrency
    verdict = {
        "H-K1": "falsified" if p["K1_rel_gap"] <= 0.10 else "supported",
        "H-K3": {arch: {k: ("in [0.85,1.15]" if 0.85 <= p[arch][k] <= 1.15 else "outside") for k in ("K3_prune_ratio", "K3_down_ratio")}
                 for arch in ("dense", "hybrid")},
        "H-K1b": {arch: ("same (<=10%)" if abs(p[arch]["K1b_down_over_prune"] - 1) <= 0.10 else "differs") for arch in ("dense", "hybrid")},
        "H-K4": {arch: ("no effect (<=10%)" if p[arch]["K4_spread"] <= 0.10 else "differs") for arch in ("dense", "hybrid")},
    }
    out = {"replaced_points": replaced, "table": table, "hypotheses_by_conc": hyp, "verdict_c128": verdict}
    json.dump(out, open(a.out, "w"), indent=1)

    print("replaced:", replaced)
    for m in MODELS:
        print(f"\n{m}")
        print("method      conc  running  gain[min,max]          req/s  req_gain  kv    preempt  ttft50  tpot50")
        for me in METHODS:
            for c in CONCS:
                r = table[f"{m}|{me}|{c}"]
                g = r["running_gain"]
                print(f"{me:10s} {c:4d}  {r['running']:6.2f}  {g[0]:.3f}[{g[1]:.3f},{g[2]:.3f}]  {r['req_s']:5.2f}  "
                      f"{r['req_s_gain'][0]:.3f}    {r['kv_usage']:.3f} {r['preemptions']:6.0f}  {r['ttft_p50']:7.0f} {r['tpot_p50']:6.1f}")
    print("\nhypotheses by concurrency")
    for c in CONCS:
        h = hyp[c]
        print(c, {k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in h.items()})
    print("\nverdict at c128:", json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
