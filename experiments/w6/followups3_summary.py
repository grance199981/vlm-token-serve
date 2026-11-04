"""Follow-up U1 (docs/plans/FOLLOWUP3_PLAN.md): GPU clocks / power during each measured point, next to the
ViT time per image, and the pre-registered verdicts.

Usage: python followups3_summary.py --run <run_dir> [--json out.json]
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import statistics as st

SERVE = re.compile(r"^(?P<model>.+?)_(?P<w>[PD])_(?P<method>[a-z0-9_]+?)_k(?P<keep>[0-9.]+)_b(?P<b>\w+)"
                   r"_c(?P<c>\d+)_rep(?P<rep>\d+)\.json$")
MODELS = ["Qwen3-VL-4B-Instruct", "Qwen3.5-4B"]


def dmon_rows(path):
    cols, rows = None, []
    for line in open(path):
        if line.startswith("#Date"):
            cols = line[1:].split()
            continue
        if line.startswith("#") or not line.strip() or cols is None:
            continue
        v = line.split()
        if len(v) != len(cols):
            continue
        r = dict(zip(cols, v))
        t = dt.datetime.strptime(r["Date"] + r["Time"], "%Y%m%d%H:%M:%S").timestamp()
        rows.append((t, r))
    return rows


def mean_of(rows, key):
    vals = [float(r[key]) for _, r in rows if r.get(key, "-") not in ("-", "")]
    return st.mean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    d = os.path.join(a.run, "u1")
    pts = {}
    for f in sorted(glob.glob(os.path.join(d, "results", "*.json"))):
        m = SERVE.match(os.path.basename(f))
        if not m:
            continue
        g = m.groupdict()
        srv = f"{g['model']}_{g['method']}_k{g['keep']}_b{g['b']}_rep{g['rep']}"
        w = open(f[:-5] + ".window").read().split()
        w0, w1 = float(w[0]), float(w[2])
        drows = [x for x in dmon_rows(os.path.join(d, f"dmon_{srv}.log")) if w0 <= x[0] <= w1]
        calls = []
        for tf in glob.glob(os.path.join(d, "vit", srv, "vit_*.jsonl")):
            for line in open(tf):
                c = json.loads(line)
                if w0 <= c["t"] <= w1:
                    calls.append(c)
        imgs = sum(c["images"] for c in calls)
        b = json.load(open(f))
        pts[(g["model"], g["method"], int(g["c"]))] = {
            "req_s": b["request_throughput"], "samples": len(drows),
            "sm_clk": mean_of(drows, "pclk"), "mem_clk": mean_of(drows, "mclk"), "power": mean_of(drows, "pwr"),
            "temp": mean_of(drows, "gtemp"), "util": mean_of(drows, "sm"), "pviol": mean_of(drows, "pviol"),
            "tviol": mean_of(drows, "tviol"), "vit_ms_img": sum(c["ms"] for c in calls) / imgs if imgs else float("nan"),
            "img_per_call": imgs / len(calls) if calls else float("nan")}
    print("| model | config | conc | req/s | dmon samples | SM clock MHz | mem clock | power W | temp C | SM util % | power-limit % | ViT ms / image | images / call |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, p in sorted(pts.items()):
        print(f"| {k[0]} | {k[1]} | {k[2]} | {p['req_s']:.3f} | {p['samples']} | {p['sm_clk']:.0f} | {p['mem_clk']:.0f} | "
              f"{p['power']:.0f} | {p['temp']:.0f} | {p['util']:.0f} | {p['pviol']:.0f} | {p['vit_ms_img']:.2f} | {p['img_per_call']:.2f} |")
    out = {"points": {"/".join(map(str, k)): v for k, v in pts.items()}, "verdict": {}}
    ua, ub, uc = [], [], []
    for model in MODELS:
        b1, b96 = pts.get((model, "none", 1)), pts.get((model, "none", 96))
        f96 = pts.get((model, "fp8none", 96))
        if not (b1 and b96 and f96):
            continue
        clk_drop = 1 - b96["sm_clk"] / b1["sm_clk"]
        vit_ratio = b96["vit_ms_img"] / b1["vit_ms_img"]
        inv_clk = b1["sm_clk"] / b96["sm_clk"]
        fp8_vs_bf16_clk = f96["sm_clk"] / b96["sm_clk"] - 1
        a_, b_, c_ = clk_drop >= 0.05, abs(vit_ratio / inv_clk - 1) <= 0.05, fp8_vs_bf16_clk >= 0.03
        ua.append(a_)
        ub.append(b_)
        uc.append(c_)
        out["verdict"][model] = {"clk_drop": clk_drop, "vit_ratio": vit_ratio, "inv_clk_ratio": inv_clk,
                                 "fp8_minus_bf16_clk": fp8_vs_bf16_clk, "U1a": a_, "U1b": b_, "U1c": c_}
        print(f"{model}: bf16 SM clock 1 -> 96 drops {100 * clk_drop:.1f}% (U1a cell {'holds' if a_ else 'falsified'}); "
              f"bf16 ViT ratio 96/1 = {vit_ratio:.3f} vs clock1/clock96 = {inv_clk:.3f} (U1b cell {'holds' if b_ else 'falsified'}); "
              f"FP8 vs bf16 SM clock at 96: {100 * fp8_vs_bf16_clk:+.1f}% (U1c cell {'holds' if c_ else 'falsified'})")
    if ua:
        out["verdict"]["H-U1a"], out["verdict"]["H-U1b"], out["verdict"]["H-U1c"] = all(ua), all(ub), all(uc)
        print(f"H-U1a {'holds' if all(ua) else 'falsified'}; H-U1b {'holds' if all(ub) else 'falsified'}; "
              f"H-U1c {'holds' if all(uc) else 'falsified'}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
