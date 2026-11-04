"""K2 summary (docs/plans/K2_PLAN.md): per point, turn 2-4 prefix-cache hit rate (cached / prompt tokens,
summed), share of turn 2-4 requests with any hit, turn 2-4 TTFT median, turn-1 prompt tokens; the
capacity threshold S* (hit rate falls to half of its S = 8 value, linear between grid points); and
the pre-registered verdicts when --pred (the §5 predictions, JSON {model: {method: S*}}) is given.

Usage: python k2_summary.py --run <run_dir> [--pred pred.json] [--json out.json]
"""
import argparse
import glob
import json
import os
import re
import statistics as st

NAME = re.compile(r"^(?P<model>.+?)_(?P<method>[a-z0-9_]+)_s(?P<s>\d+)_rep(?P<rep>\d+)\.jsonl$")
DENSE, HYBRID = "Qwen3-VL-4B-Instruct", "Qwen3.5-4B"


def point_stats(path):
    rows = [json.loads(x) for x in open(path)]
    later = [r for r in rows if r["turn"] >= 2 and r["error"] is None and r["prompt_tokens"]]
    first = [r["prompt_tokens"] for r in rows if r["turn"] == 1 and r["prompt_tokens"]]
    hit = sum(r["cached_tokens"] or 0 for r in later) / max(1, sum(r["prompt_tokens"] for r in later))
    ttft = [r["ttft_ms"] for r in later if r["ttft_ms"] is not None]
    return {"requests": len(rows), "errors": sum(r["error"] is not None for r in rows), "hit": hit,
            "any_hit": sum((r["cached_tokens"] or 0) > 0 for r in later) / max(1, len(later)),
            "ttft_med": st.median(ttft) if ttft else float("nan"),
            "turn1_prompt": st.mean(first) if first else float("nan"),
            "cached_by_turn": {t: st.mean([r["cached_tokens"] or 0 for r in later if r["turn"] == t] or [0])
                               for t in sorted({r["turn"] for r in later})}}


def s_star(curve):
    """curve: sorted [(S, hit)]; first S where hit <= half of hit at the smallest S, interpolated."""
    (s0, h0), half = curve[0], curve[0][1] / 2
    if h0 <= 0:
        return float("nan")
    for (sa, ha), (sb, hb) in zip(curve, curve[1:]):
        if hb <= half:  # ha > half here, or the previous pair would have returned
            return sa + (ha - half) / (ha - hb) * (sb - sa)
    return float("inf")  # never fell to half inside the grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pred")
    ap.add_argument("--json")
    a = ap.parse_args()
    pts = {}
    for f in sorted(glob.glob(os.path.join(a.run, "results", "*.jsonl"))):
        m = NAME.match(os.path.basename(f))
        if m:
            g = m.groupdict()
            pts[(g["model"], g["method"], int(g["rep"]), int(g["s"]))] = point_stats(f)
    print("| model | method | rep | S | requests | errors | turn-1 prompt | hit rate t2-4 | any hit | TTFT med ms | mean cached by turn |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, p in sorted(pts.items()):
        cb = " / ".join(f"{t}:{v:.0f}" for t, v in p["cached_by_turn"].items())
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {p['requests']} | {p['errors']} | {p['turn1_prompt']:.0f} | "
              f"{100 * p['hit']:.1f}% | {100 * p['any_hit']:.0f}% | {p['ttft_med']:.0f} | {cb} |")
    stars = {}
    for model, method, rep in sorted({k[:3] for k in pts}):
        curve = sorted((k[3], p["hit"]) for k, p in pts.items() if k[:3] == (model, method, rep))
        stars.setdefault((model, method), []).append(s_star(curve))
    out = {"points": {"/".join(map(str, k)): v for k, v in pts.items()}, "s_star": {}, "verdict": {}}
    print("\n| model | method | S* per rep | S* mean | gain vs B0 |")
    print("|---|---|---|---|---|")
    mean = {k: st.mean(v) for k, v in stars.items()}
    for (model, method), v in sorted(stars.items()):
        b0 = mean.get((model, "none"))
        gain = mean[(model, method)] / b0 if b0 else float("nan")
        out["s_star"][f"{model}/{method}"] = {"reps": v, "mean": mean[(model, method)], "gain": gain}
        print(f"| {model} | {method} | {', '.join(f'{x:.1f}' for x in v)} | {mean[(model, method)]:.1f} | {gain:.3f} |")
    h8 = [p["hit"] for k, p in pts.items() if k[0] == HYBRID and k[1] == "none" and k[3] == 8]
    if h8:
        c = st.mean(h8) >= 0.20
        out["verdict"]["H-K2c"] = c
        print(f"\nH-K2c: Qwen3.5 B0 S=8 hit rate {100 * st.mean(h8):.1f}% -> {'holds' if c else 'falsified'}")
    for method in sorted({k[1] for k in mean} - {"none"}):
        gd = out["s_star"].get(f"{DENSE}/{method}", {}).get("gain")
        gh = out["s_star"].get(f"{HYBRID}/{method}", {}).get("gain")
        if gd and gh:
            ok = gd / gh - 1 >= 0.10
            out["verdict"][f"H-K2a/{method}"] = ok
            print(f"H-K2a {method}: dense gain {gd:.3f} vs hybrid {gh:.3f} ({100 * (gd / gh - 1):+.1f}%) -> "
                  f"{'holds' if ok else 'falsified'}")
    if a.pred:
        pred = json.load(open(a.pred))
        cells = []
        for model, by in pred.items():
            for method, ps in by.items():
                got = mean.get((model, method))
                if got is None:
                    continue
                ok = abs(got / ps - 1) <= 0.20
                cells.append(ok)
                out["verdict"][f"H-K2b/{model}/{method}"] = {"pred": ps, "measured": got, "holds": ok}
                print(f"H-K2b {model}/{method}: predicted S* {ps:.1f}, measured {got:.1f} "
                      f"({100 * (got / ps - 1):+.1f}%) -> {'holds' if ok else 'falsified'}")
        if cells:
            out["verdict"]["H-K2b"] = all(cells)
            print(f"H-K2b overall {'holds' if all(cells) else 'falsified'} ({sum(cells)}/{len(cells)} cells)")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
