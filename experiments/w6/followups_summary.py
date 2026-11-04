"""Summaries for follow-ups S1-S4 (docs/plans/FOLLOWUP_PLAN.md).

Usage: python followups_summary.py --run <run_dir> [--json out.json]
  s1a / s1b: K1 steady running concurrency gain vs B0 (paired per rep), vs the locked predictions
  s2: ViT trace per server (time share, images per call, ms per image) and the Amdahl prediction
  s3: Qwen3.5 D-shape prefill, "D only" vs "P then D", prefix caching off / on
  s4: B0 throughput per concurrency, images encoded per request, scheduler metrics
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w4"))
from k1_steady import steady  # noqa: E402

PRED = {  # docs/plans/FOLLOWUP_PLAN.md S1 (k1_predict.py --m 1, --text 76 / 80)
    "s1a": {"Qwen3-VL-4B-Instruct": {"random": 2.947, "downsample": 2.947},
            "Qwen3.5-4B": {"random": 1.464, "downsample": 1.464}},
    "s1b": {"Qwen3-VL-4B-Instruct": {"random": 1.893, "downsample": 1.859},
            "Qwen3.5-4B": {"random": 1.522, "downsample": 1.502}},
}
MODELS = ["Qwen3-VL-4B-Instruct", "Qwen3.5-4B"]
SERVE = re.compile(r"^(?P<model>.+?)_(?P<w>[PD])_(?P<method>[a-z0-9_]+?)_k(?P<keep>[0-9.]+)_b(?P<b>\w+)"
                   r"_c(?P<c>\d+)_rep(?P<rep>\d+)\.json$")


def s1(run, name, out):
    res = os.path.join(run, name, "results")
    if not os.path.isdir(res):
        return
    rows = {}
    for f in glob.glob(os.path.join(res, "*_c128_rep*.json")):
        m = re.match(r"^(.+?)_(none|random|downsample)_c128_rep(\d+)\.json$", os.path.basename(f))
        b = json.load(open(f))
        s = steady(f[:-5] + ".metrics.csv")
        rows[(m.group(1), m.group(2), int(m.group(3)))] = {"running": s["running"], "kv": s["kv_usage"],
                                                           "failed": b["failed"], "req_s": b["request_throughput"]}
    print(f"\n## {name}: steady running concurrency gain at conc 128 (mean over reps [min, max])")
    print("| model | method | B0 running | running | gain | predicted | measured / predicted | KV usage |")
    print("|---|---|---|---|---|---|---|---|")
    out[name] = {}
    for model in MODELS:
        for meth in ("random", "downsample"):
            reps = sorted(r for (mm, me, r) in rows if mm == model and me == meth and (mm, "none", r) in rows)
            if not reps:
                continue
            g = [rows[(model, meth, r)]["running"] / rows[(model, "none", r)]["running"] for r in reps]
            pred = PRED[name][model][meth]
            mean = sum(g) / len(g)
            out[name][f"{model}/{meth}"] = {"gain": mean, "range": [min(g), max(g)], "pred": pred,
                                            "ratio": mean / pred, "reps": reps}
            b0 = st.mean(rows[(model, "none", r)]["running"] for r in reps)
            run_m = st.mean(rows[(model, meth, r)]["running"] for r in reps)
            kv = st.mean(rows[(model, meth, r)]["kv"] for r in reps)
            print(f"| {model} | {meth} | {b0:.2f} | {run_m:.2f} | {mean:.3f} [{min(g):.3f}, {max(g):.3f}] | "
                  f"{pred:.3f} | {mean / pred:.3f} | {kv:.2f} |")
    d = out[name]
    for meth in ("random", "downsample"):
        a, b = d.get(f"{MODELS[0]}/{meth}"), d.get(f"{MODELS[1]}/{meth}")
        if a and b:
            gap = (a["gain"] - b["gain"]) / a["gain"]
            d[f"gap/{meth}"] = gap
            print(f"structure gap ({meth}): {100 * gap:.1f}%")


def window(path):
    lines = open(path).read().split()
    return float(lines[0]), float(lines[2])


def trace_stats(run, srv, w0, w1):
    calls = []
    for f in glob.glob(os.path.join(run, "vit", srv, "vit_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if w0 <= r["t"] <= w1:
                calls.append(r)
    ms = sum(c["ms"] for c in calls)
    imgs = sum(c["images"] for c in calls)
    return {"calls": len(calls), "ms": ms, "images": imgs,
            "img_per_call": imgs / len(calls) if calls else float("nan"),
            "ms_per_img": ms / imgs if imgs else float("nan")}


def serve_points(run, name):
    res = os.path.join(run, name, "results")
    pts = {}
    for f in sorted(glob.glob(os.path.join(res, "*.json"))):
        m = SERVE.match(os.path.basename(f))
        if not m:
            continue
        g = m.groupdict()
        b = json.load(open(f))
        w0, w1 = window(f[:-5] + ".window")
        srv = f"{g['model']}_{g['method']}_k{g['keep']}_b{g['b']}_rep{g['rep']}"
        t = trace_stats(os.path.join(run, name), srv, w0, w1)
        p = {"req_s": b["request_throughput"], "completed": b["completed"], "failed": b["failed"],
             "duration": b["duration"], **t}
        p["T_ms"] = 1000 * b["duration"] / b["completed"]
        p["V_ms"] = t["ms"] / b["completed"]
        p["img_per_req"] = t["images"] / b["completed"]
        mcsv = f[:-5] + ".metrics.csv"
        if os.path.exists(mcsv):
            p.update({f"m_{k}": v for k, v in steady(mcsv).items()})
        pts[(g["model"], g["method"], int(g["c"]), int(g["rep"]))] = p
    return pts


def s2(run, out):
    if not os.path.isdir(os.path.join(run, "s2")):
        return
    pts = serve_points(run, "s2")
    print("\n## s2: ViT trace, P workload, conc 64")
    print("| model | config | req/s | T ms/req | ViT ms/req | ViT share | images / ViT call | ViT ms / image |")
    print("|---|---|---|---|---|---|---|---|")
    out["s2"] = {}
    for (model, meth, c, rep), p in sorted(pts.items()):
        out["s2"][f"{model}/{meth}"] = p
        print(f"| {model} | {meth} | {p['req_s']:.3f} | {p['T_ms']:.1f} | {p['V_ms']:.1f} | "
              f"{p['V_ms'] / p['T_ms']:.3f} | {p['img_per_call']:.1f} | {p['ms_per_img']:.2f} |")
    for model in MODELS:
        g = lambda meth: pts.get((model, meth, 64, 1))  # noqa: E731
        b0, rt, lb0, lrt = g("none"), g("rate"), g("fp8lnone"), g("fp8lrate")
        if not all((b0, rt, lb0, lrt)):
            continue
        # T = per-request time (1 / throughput), V = ViT time per request, L = T - V
        L = lambda p: p["T_ms"] - p["V_ms"]  # noqa: E731
        s = L(b0) / L(lb0)  # LLM-part speedup of LLM-only FP8, from B0
        t_pred = lrt["V_ms"] + L(rt) / s  # Amdahl: RATE's non-ViT part sped up by the same s
        g_pred = rt["T_ms"] / t_pred  # FP8 (LLM only) speedup on the RATE config
        g_meas = rt["T_ms"] / lrt["T_ms"]
        # R = g(fp8l+rate) / (g(fp8l) * g(rate)) = T_l(B0) * T_b(rate) / (T_l(rate) * T_b(B0))
        R_meas = lb0["T_ms"] * rt["T_ms"] / (lrt["T_ms"] * b0["T_ms"])
        R_pred = lb0["T_ms"] * rt["T_ms"] / (t_pred * b0["T_ms"])
        out["s2"][f"{model}/amdahl"] = {"s_llm": s, "g_pred_rate": g_pred, "g_meas_rate": g_meas,
                                        "R_pred": R_pred, "R_meas": R_meas}
        print(f"{model}: LLM speedup s = {s:.3f}; FP8(LLM-only) gain on RATE predicted {g_pred:.3f} vs measured "
              f"{g_meas:.3f}; R predicted {R_pred:.3f} vs measured {R_meas:.3f} (traced runs)")


def s3(run, out):
    d = os.path.join(run, "s3", "Qwen3.5-4B")
    if not os.path.isdir(d):
        return
    print("\n## s3: Qwen3.5 D-shape (1 image) LLM prefill ms, per process")
    out["s3"] = {}
    for pc in ("off", "on"):
        vals = {}
        for mode in ("D", "PD"):
            v = []
            for f in sorted(glob.glob(os.path.join(d, f"{mode}_r*_pc{pc}.json"))):
                v.append(json.load(open(f))["shapes"]["D"]["llm_prefill_ms"])
            vals[mode] = v
        if vals["D"] and vals["PD"]:
            diff = st.median(vals["PD"]) - st.median(vals["D"])
            out["s3"][pc] = {"D": vals["D"], "PD": vals["PD"], "diff": diff}
            print(f"prefix caching {pc}: D only {[round(x, 1) for x in vals['D']]}, P then D "
                  f"{[round(x, 1) for x in vals['PD']]}; median difference {diff:+.1f} ms")


def s4(run, out):
    if not os.path.isdir(os.path.join(run, "s4")):
        return
    pts = serve_points(run, "s4")
    print("\n## s4: B0, P workload, budget 2048")
    print("| model | conc | rep | req/s | images encoded / request | ViT calls | running | waiting | KV usage | preemptions |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    out["s4"] = {}
    for (model, meth, c, rep), p in sorted(pts.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][3])):
        out["s4"][f"{model}/c{c}/rep{rep}"] = p
        print(f"| {model} | {c} | {rep} | {p['req_s']:.3f} | {p['img_per_req']:.3f} | {p['calls']} | "
              f"{p.get('m_running', float('nan')):.1f} | {p.get('m_waiting', float('nan')):.1f} | "
              f"{p.get('m_kv_usage', float('nan')):.2f} | {p.get('m_preemptions', float('nan')):.0f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for name in ("s1a", "s1b"):
        s1(a.run, name, out)
    s2(a.run, out)
    s3(a.run, out)
    s4(a.run, out)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
