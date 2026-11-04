"""Summarize `vllm bench serve` JSON results from run_serving.sh.

Files are named <model>_<workload>_c<conc>_rep<k>.json. For each (model, workload,
conc) prints the median over repetitions and the min-max range, for request
throughput, output token throughput, TTFT p50/p99 and TPOT p50/p99.

Usage: python experiments/w1/summarize_serving.py <results_dir> [--csv out.csv]
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict

PAT = re.compile(r"^(?P<model>.+)_(?P<wl>[PD])_c(?P<conc>\d+)_rep(?P<rep>\d+)\.json$")
FIELDS = [("request_throughput", "req/s"), ("output_throughput", "out tok/s"),
          ("p50_ttft_ms", "TTFT p50"), ("p99_ttft_ms", "TTFT p99"),
          ("p50_tpot_ms", "TPOT p50"), ("p99_tpot_ms", "TPOT p99")]


def get(d, key):
    if key in d:
        return d[key]
    alt = key.replace("p50_", "median_")
    return d.get(alt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--csv")
    args = ap.parse_args()
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(args.results_dir, "*.json")):
        m = PAT.match(os.path.basename(path))
        if not m:
            continue
        d = json.load(open(path, encoding="utf-8"))
        if d.get("failed", 0):
            print(f"WARNING {os.path.basename(path)}: failed={d['failed']} completed={d.get('completed')}")
        groups[(m["model"], m["wl"], int(m["conc"]))].append(d)
    rows = []
    for (model, wl, conc), ds in sorted(groups.items()):
        row = {"model": model, "workload": wl, "conc": conc, "reps": len(ds)}
        for key, _ in FIELDS:
            vals = [v for v in (get(d, key) for d in ds) if v is not None]
            if vals:
                row[key] = statistics.median(vals)
                row[key + "_min"], row[key + "_max"] = min(vals), max(vals)
        rows.append(row)
    hdr = "| model | W | conc | reps | " + " | ".join(lbl for _, lbl in FIELDS) + " |"
    print(hdr)
    print("|" + "---|" * (4 + len(FIELDS)))
    for r in rows:
        cells = []
        for key, _ in FIELDS:
            if key in r:
                spread = (r[key + "_max"] - r[key + "_min"]) / r[key] * 100 if r[key] else 0
                cells.append(f"{r[key]:.1f} (±{spread / 2:.0f}%)")
            else:
                cells.append("—")
        print(f"| {r['model']} | {r['workload']} | {r['conc']} | {r['reps']} | " + " | ".join(cells) + " |")
    if args.csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "workload", "conc", "reps"), k))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
