"""Summaries and pre-registered verdicts for follow-ups T1-T5 (docs/plans/FOLLOWUP2_PLAN.md).

Usage: python followups2_summary.py --run <run_dir> [--json out.json]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w4"))
from k1_steady import steady  # noqa: E402

SERVE = re.compile(r"^(?P<model>.+?)_(?P<w>[PD])_(?P<method>[a-z0-9_]+?)_k(?P<keep>[0-9.]+)_b(?P<b>\w+)"
                   r"_c(?P<c>\d+)_rep(?P<rep>\d+)\.json$")
MODELS = ["Qwen3-VL-4B-Instruct", "Qwen3.5-4B"]


def window(path):
    v = open(path).read().split()
    return float(v[0]), float(v[2])


def lines(pattern, w0, w1):
    out = []
    for f in glob.glob(pattern):
        for line in open(f):
            r = json.loads(line)
            if w0 <= r["t"] <= w1:
                out.append(r)
    return out


def points(stage_dir):
    res = os.path.join(stage_dir, "results")
    for f in sorted(glob.glob(os.path.join(res, "*.json"))):
        m = SERVE.match(os.path.basename(f))
        if m:
            g = m.groupdict()
            srv = f"{g['model']}_{g['method']}_k{g['keep']}_b{g['b']}_rep{g['rep']}"
            yield g, srv, json.load(open(f)), window(f[:-5] + ".window")


def t1(run, out):
    d = os.path.join(run, "t1")
    if not os.path.isdir(d):
        return
    rows = {}
    for g, srv, b, (w0, w1) in points(d):
        steps = lines(os.path.join(d, "steps", srv, "steps_*.jsonl"), w0, w1)
        if not steps:
            continue
        dur = w1 - w0
        rows[(g["model"], int(g["c"]), int(g["rep"]))] = {
            "req_s": b["request_throughput"], "steps": len(steps),
            "tok_per_step": st.mean(s["tokens"] for s in steps), "wall_ms": st.mean(s["wall"] for s in steps),
            "cpu_ms": st.mean(s["sched"] + s["update"] + s["gc"] for s in steps),
            "sched_ms": st.mean(s["sched"] for s in steps), "update_ms": st.mean(s["update"] for s in steps),
            "gc_ms": st.mean(s["gc"] for s in steps), "busy": sum(s["wall"] for s in steps) / 1000 / dur,
            "waiting": st.mean(s["waiting"] for s in steps), "running": st.mean(s["running"] for s in steps)}
    print("\n## T1: B0, P workload, budget 2048, engine step trace (per point)")
    print("| model | conc | rep | req/s | steps | tokens/step | step ms | CPU ms (sched / update / gc) | busy share | waiting | running |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in sorted(rows.items()):
        print(f"| {k[0]} | {k[1]} | {k[2]} | {r['req_s']:.3f} | {r['steps']} | {r['tok_per_step']:.0f} | {r['wall_ms']:.1f} | "
              f"{r['cpu_ms']:.2f} ({r['sched_ms']:.2f} / {r['update_ms']:.2f} / {r['gc_ms']:.2f}) | {r['busy']:.3f} | "
              f"{r['waiting']:.1f} | {r['running']:.1f} |")
    out["t1"] = {"points": {f"{k[0]}/c{k[1]}/rep{k[2]}": v for k, v in rows.items()}, "verdict": {}}
    for model in MODELS:
        reps = sorted({k[2] for k in rows if k[0] == model and (model, 32, k[2]) in rows and (model, 96, k[2]) in rows})
        if not reps:
            continue
        comp = {"tokens": [], "busy": [], "cpu": [], "rest": [], "thr": []}
        for r in reps:
            a, z = rows[(model, 32, r)], rows[(model, 96, r)]
            wbar = (a["wall_ms"] + z["wall_ms"]) / 2
            comp["thr"].append(math.log(z["req_s"] / a["req_s"]))
            comp["tokens"].append(math.log(z["tok_per_step"] / a["tok_per_step"]))
            comp["busy"].append(math.log(z["busy"] / a["busy"]))
            comp["cpu"].append(-(z["cpu_ms"] - a["cpu_ms"]) / wbar)
            comp["rest"].append(-((z["wall_ms"] - z["cpu_ms"]) - (a["wall_ms"] - a["cpu_ms"])) / wbar)
        m = {k: st.mean(v) for k, v in comp.items()}
        shares = {k: (m[k] / m["thr"] if m["thr"] else float("nan")) for k in ("tokens", "busy", "cpu", "rest")}
        drop = 1 - math.exp(m["thr"])
        top = max(shares, key=lambda k: shares[k])
        verdict_a = shares[top] >= 0.5 and top in ("tokens", "cpu", "rest")
        out["t1"]["verdict"][model] = {"dln": m, "shares": shares, "drop_32_96": drop, "largest": top,
                                       "H-T1a": verdict_a, "H-T1b": drop >= 0.03}
        print(f"{model}: throughput 32 -> 96 drops {100 * drop:.1f}% (H-T1b {'holds' if drop >= 0.03 else 'falsified'}); "
              f"shares of the drop: tokens/step {shares['tokens']:.2f}, engine busy share {shares['busy']:.2f}, "
              f"CPU in step {shares['cpu']:.2f}, rest of step {shares['rest']:.2f}; largest = {top}; "
              f"H-T1a {'holds' if verdict_a else 'falsified'}")


def vit_table(stage_dir):
    rows = {}
    for g, srv, b, (w0, w1) in points(stage_dir):
        calls = lines(os.path.join(stage_dir, "vit", srv, "vit_*.jsonl"), w0, w1)
        imgs = sum(c["images"] for c in calls)
        ms = sum(c["ms"] for c in calls)
        rows[(g["model"], g["method"])] = {"calls": len(calls), "images": imgs,
                                           "img_per_call": imgs / len(calls) if calls else float("nan"),
                                           "ms_per_img": ms / imgs if imgs else float("nan")}
    return rows


def t2(run, out):
    out["t2"] = {}
    for name, wl in (("t2p", "P"), ("t2d", "D")):
        d = os.path.join(run, name)
        if not os.path.isdir(d):
            continue
        rows = vit_table(d)
        print(f"\n## T2 ({wl} workload, conc 1): ViT per image, bf16 vs full FP8")
        print("| model | config | ViT calls | images / call | ms / image |")
        print("|---|---|---|---|---|")
        for (model, meth), r in sorted(rows.items()):
            print(f"| {model} | {meth} | {r['calls']} | {r['img_per_call']:.2f} | {r['ms_per_img']:.2f} |")
        for model in MODELS:
            if (model, "none") in rows and (model, "fp8none") in rows:
                ratio = rows[(model, "fp8none")]["ms_per_img"] / rows[(model, "none")]["ms_per_img"]
                out["t2"][f"{wl}/{model}"] = {"ratio_fp8_bf16": ratio, "holds": ratio <= 1.05}
                print(f"{model}: FP8 / bf16 ms per image = {ratio:.3f} -> H-T2 cell {'holds' if ratio <= 1.05 else 'falsified'}")


def t3(run, out):
    d = os.path.join(run, "t3")
    if not os.path.isdir(d):
        return
    print("\n## T3: text-only prefill (prefix caching on), LLM forward ms")
    print("| model | length | bf16 | FP8 (LLM only) | speedup |")
    print("|---|---|---|---|---|")
    out["t3"] = {}
    for model in MODELS:
        fb, fq = os.path.join(d, model, "pc_on.json"), os.path.join(d, model, "pc_on_fp8l.json")
        if not (os.path.exists(fb) and os.path.exists(fq)):
            continue
        b, q = json.load(open(fb))["lengths"], json.load(open(fq))["lengths"]
        sp = {}
        for L in sorted(b, key=int):
            sp[L] = b[L]["llm_forward_ms"] / q[L]["llm_forward_ms"]
            print(f"| {model} | {L} | {b[L]['llm_forward_ms']:.1f} | {q[L]['llm_forward_ms']:.1f} | {sp[L]:.3f} |")
        if "1612" in sp and "3180" in sp:
            out["t3"][model] = {"speedup": sp, "ratio_1612_3180": sp["1612"] / sp["3180"]}
            print(f"{model}: speedup(1612) / speedup(3180) = {sp['1612'] / sp['3180']:.3f}")
    a, z = out["t3"].get("Qwen3.5-4B"), out["t3"].get("Qwen3-VL-4B-Instruct")
    if a and z:
        holds = a["ratio_1612_3180"] <= 0.95 and z["ratio_1612_3180"] >= 0.97
        out["t3"]["H-T3"] = holds
        print(f"H-T3 {'holds' if holds else 'falsified'}")


def t4(run, out):
    d = os.path.join(run, "t4", "Qwen3.5-4B")
    if not os.path.isdir(d):
        return

    def get(pat):
        return [json.load(open(f))["shapes"]["D"]["llm_prefill_ms"] for f in sorted(glob.glob(os.path.join(d, pat)))]

    dd, pd, pr = get("D_r*_pcon.json"), get("PD_r*_pcon.json"), get("PD_r*_pcon_reset.json")
    print("\n## T4: Qwen3.5 D-shape LLM prefill ms per process (prefix caching on)")
    print(f"D only {[round(x, 1) for x in dd]}; P then D {[round(x, 1) for x in pd]}; P, reset, D {[round(x, 1) for x in pr]}")
    if dd and pd and pr:
        a = st.median(pd) - st.median(dd)
        b = st.median(pr) - st.median(dd)
        out["t4"] = {"D": dd, "PD": pd, "PD_reset": pr, "diff_PD": a, "diff_reset": b,
                     "H-T4a": a >= 8, "H-T4b": (b <= 3) if a >= 8 else None}
        print(f"P then D - D only = {a:+.1f} ms (H-T4a {'holds' if a >= 8 else 'falsified'}); "
              f"reset - D only = {b:+.1f} ms (H-T4b {'holds' if b <= 3 else 'falsified'}{'' if a >= 8 else ', not judged'})")


def t5(run, out):
    rows = {}
    for o in ("456", "480"):
        for f in sorted(glob.glob(os.path.join(run, f"t5_o{o}", "results", "*_none_c*_rep*.json"))):
            rep = int(re.search(r"_rep(\d+)\.json$", f).group(1))
            s = steady(f[:-5] + ".metrics.csv")
            b = json.load(open(f))
            rows[(o, rep)] = {"running": s["running"], "preemptions": s["preemptions"], "kv": s["kv_usage"],
                              "req_s": b["request_throughput"], "prompt": b["total_input_tokens"] / b["completed"]}
    if not rows:
        return
    print("\n## T5: Qwen3.5 B0, K1 setting, output 456 vs 480")
    print("| output | rep | prompt | running | preemptions | KV usage | req/s |")
    print("|---|---|---|---|---|---|---|")
    for (o, rep), r in sorted(rows.items()):
        print(f"| {o} | {rep} | {r['prompt']:.0f} | {r['running']:.2f} | {r['preemptions']:.0f} | {r['kv']:.3f} | {r['req_s']:.3f} |")
    reps = sorted({k[1] for k in rows if ("456", k[1]) in rows and ("480", k[1]) in rows})
    if reps:
        pre = st.mean(rows[("480", r)]["preemptions"] for r in reps) / max(1e-9, st.mean(rows[("456", r)]["preemptions"] for r in reps))
        run_ratio = st.mean(rows[("480", r)]["running"] / rows[("456", r)]["running"] for r in reps)
        holds = pre >= 3 and run_ratio <= 0.97
        out["t5"] = {"rows": {f"{k[0]}/rep{k[1]}": v for k, v in rows.items()}, "preempt_ratio": pre,
                     "running_ratio": run_ratio, "H-T5": holds}
        print(f"preemptions 480 / 456 = {pre:.2f}; running 480 / 456 = {run_ratio:.3f} (ledger predicts 0.995) -> "
              f"H-T5 {'holds' if holds else 'falsified'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for f in (t1, t2, t3, t4, t5):
        f(a.run, out)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
