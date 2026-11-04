"""Steady-state summary of one K1 metrics CSV (docs/plans/W4_PLAN.md §4.1): drop the first 20%
and last 10% of the samples, report mean running / waiting / KV usage and preemptions.

Usage: python k1_steady.py <metrics.csv> [--json]
"""
import csv
import json
import sys


def steady(path):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 10:
        return {"n": len(rows)}
    seg = rows[int(0.2 * len(rows)):int(0.9 * len(rows))]

    def mean(k):
        return sum(float(r[k]) for r in seg) / len(seg)

    return {"n": len(seg), "running": mean("running"), "waiting": mean("waiting"),
            "kv_usage": mean("kv_usage"),
            "preemptions": float(rows[-1]["preemptions"]) - float(rows[0]["preemptions"])}


if __name__ == "__main__":
    s = steady(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(s))
    else:
        print("steady " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
