"""Join bench results with the dmon / CPU probes from run_serving.sh.

For every point, the window is [end - duration, end], where `date` in the bench
JSON is the save time (end). Reports median GPU SM% (nvidia-smi dmon, 1 s) and
median CPU% of the API server and EngineCore (/proc deltas, 2 s; 100 = one core),
median over the 3 repetitions.

Usage: python experiments/w1/bottleneck_probe.py <run_dir>
"""
import datetime as dt
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

PAT = re.compile(r"^(?P<model>.+)_(?P<wl>[PD])_c(?P<conc>\d+)_rep(?P<rep>\d+)\.json$")


def load_series(path, cols):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        try:
            t = dt.datetime.strptime(parts[0], "%H:%M:%S").time()
            out.append((t, [float(parts[c]) for c in cols]))
        except (ValueError, IndexError):
            continue
    return out


def window_mean(series, t0, t1, k):
    vals = [v[k] for t, v in series if t0 <= t <= t1]
    return statistics.median(vals) if vals else float("nan"), len(vals)


def main(run_dir):
    rows = defaultdict(list)
    for path in glob.glob(os.path.join(run_dir, "results", "*.json")):
        m = PAT.match(os.path.basename(path))
        if not m:
            continue
        d = json.load(open(path, encoding="utf-8"))
        end = dt.datetime.strptime(d["date"], "%Y%m%d-%H%M%S")
        start = end - dt.timedelta(seconds=d["duration"])
        if start.date() != end.date():
            continue  # window crosses midnight; HH:MM:SS logs cannot be matched safely
        dmon = load_series(os.path.join(run_dir, f"dmon_{m['model']}_rep{m['rep']}.log"), [2])
        cpu = load_series(os.path.join(run_dir, f"cpu_{m['model']}_rep{m['rep']}.log"), [1, 2])
        sm, nsm = window_mean(dmon, start.time(), end.time(), 0)
        api, ncpu = window_mean(cpu, start.time(), end.time(), 0)
        eng, _ = window_mean(cpu, start.time(), end.time(), 1)
        rows[(m["model"], m["wl"], int(m["conc"]))].append((sm, api, eng, nsm, ncpu, d["duration"]))
    print("| model | W | conc | reps | GPU SM% | API server CPU% | EngineCore CPU% | samples (dmon/cpu) |")
    print("|---|---|---|---|---|---|---|---|")
    for (model, wl, conc), rs in sorted(rows.items()):
        med = lambda i: statistics.median(r[i] for r in rs)
        print(f"| {model} | {wl} | {conc} | {len(rs)} | {med(0):.0f} | {med(1):.0f} | {med(2):.0f} | "
              f"{sum(r[3] for r in rs)}/{sum(r[4] for r in rs)} |")


if __name__ == "__main__":
    main(sys.argv[1])
