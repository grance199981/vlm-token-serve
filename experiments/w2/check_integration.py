"""Checks for the W2 integration run (T1-T3), revision 2 (2026-10-20).

Revision 1 compared greedy TEXT only and failed on Qwen3-VL because vanilla vLLM
itself is not reproducible when prefix caching changes the kernel path (bf16):
vanilla pass0 vs pass1 differed on request 10. These criteria were fixed BEFORE
the revision-2 run.

Per pair of runs (A, B) and request: the generated tokens' common prefix and the
max |delta logprob| of the chosen tokens over that prefix ("dlp").

T1a  keep_all_nopc vs vanilla_nopc (no cache, sequential): identical text and
     dlp == 0 on every request                                      [hard]
T1b  keep_all vs vanilla (with cache): every differing request is also unstable
     in vanilla (differs between vanilla passes or vs vanilla_nopc)  [hard]
T1n  negative control: random50_nopc vs vanilla_nopc text differs on >= 25%  [hard]
T2   prompt_tokens(vanilla) - prompt_tokens(method) == sum(N - keep_count(N))
     on every request                                                [hard]
T3   chunking / prefix cache do not break positions: median per-request dlp of
     (random50, random50_c256) and (random50, random50_nopc) <= max(0.05,
     3 x median dlp of the vanilla pair with the same perturbation)   [hard]
T3n  negative control: random50_shift7 (text after images shifted by 7, a
     deliberate position error) vs random50_nopc must EXCEED that threshold [hard]

Revision-2 results (kept as they were): Qwen3-VL failed T1a (texts 26/26 equal,
max dlp 0.095) and both models failed T3n (shift 7 is not detected by the median
dlp). Diagnostics added afterwards, reported separately, NOT replacing the above:
D1   keep_all_nopc vs vanilla_evs_nopc (vLLM own pruning machinery switched on
     via --video-pruning-rate, no plugin): dlp == 0 would attribute the T1a gap
     to vLLM's pruning code path rather than to the plugin
D2   random50_transpose (row/col of image tokens swapped) vs random50_nopc:
     median dlp vs the T3 threshold (is T3 sensitive to image-position errors?)
"""
import json
import math
import os
import statistics
import sys

KEEP = {"keep_all": None, "keep_all_nopc": None, "random50": 0.5, "random50_c256": 0.5,
        "random50_nopc": 0.5, "random50_shift7": 0.5, "random50_transpose": 0.5, "divprune25": 0.25, "rate25": 0.25,
        "rate_pr25": 0.25, "pool2x2": "pool"}


def kept(n, gh, gw, keep):
    if keep is None:
        return n
    if keep == "pool":
        return math.ceil(gh / 2) * math.ceil(gw / 2)
    return max(1, int(n * keep))


def load(d, model, name):
    p = os.path.join(d, f"{model}_{name}.json")
    return {(x["rep"], x["req"]): x for x in json.load(open(p))} if os.path.exists(p) else None


def dlp(x, y):
    a, b = x.get("logprobs") or [], y.get("logprobs") or []
    m = 0.0
    for (ta, la), (tb, lb) in zip(a, b):
        if ta != tb:
            break
        m = max(m, abs(la - lb))
    return m


def differing(a, b):
    return {k for k in a if a[k]["output"] != b[k]["output"]}


def med_dlp(a, b):
    return statistics.median(dlp(a[k], b[k]) for k in a)


def main(d):
    models = sorted({f.split("_")[0] for f in os.listdir(d) if f.endswith(".json")})
    ok = True
    rows = []

    def rec(m, name, res, detail, hard=True):
        nonlocal ok
        if hard:
            ok &= bool(res)
        rows.append(f"| {m} | {name} | {'PASS' if res else ('FAIL' if hard else 'INFO')} | {detail} |")

    for m in models:
        L = {n: load(d, m, n) for n in ["vanilla", "vanilla_nopc", "vanilla_c256", "vanilla_evs_nopc", *KEEP]}
        van, vn = L["vanilla"], L["vanilla_nopc"]
        if L["keep_all_nopc"] and vn:
            same = differing(vn, L["keep_all_nopc"])
            mx = max(dlp(vn[k], L["keep_all_nopc"][k]) for k in vn)
            rec(m, "T1a keep_all_nopc == vanilla_nopc", not same and mx == 0.0,
                f"text differs on {len(same)}/{len(vn)}, max dlp {mx:.2e}")
        if L["keep_all"] and van and vn:
            unstable = {k for k in van if van[k]["output"] != van[(1 - k[0], k[1])]["output"]}
            unstable |= {(r, q) for (r, q) in differing(van, vn) for r in (0, 1)}
            unstable |= {(1 - r, q) for (r, q) in unstable}
            diff = differing(van, L["keep_all"])
            rec(m, "T1b keep_all vs vanilla (cache on): diffs only on vanilla-unstable requests",
                diff <= unstable, f"differ {sorted(diff)}; vanilla-unstable {sorted(unstable)}")
        if L["random50_nopc"] and vn:
            frac = len(differing(vn, L["random50_nopc"])) / len(vn)
            rec(m, "T1n negative control: random50_nopc differs from vanilla_nopc", frac >= 0.25,
                f"{frac:.0%} of requests differ")
        if van:
            for name, keep in KEEP.items():
                run = L.get(name)
                if not run:
                    continue
                bad = sum((van[k]["prompt_tokens"] - run[k]["prompt_tokens"])
                          != sum(gh * gw - kept(gh * gw, gh, gw, keep) for gh, gw in van[k]["grids"])
                          for k in van)
                rec(m, f"T2 token count {name}", bad == 0, f"{len(van) - bad}/{len(van)} exact")
        thr_used = []
        for pert in ("c256", "nopc"):
            vp, rp, r = L.get(f"vanilla_{pert}"), L.get(f"random50_{pert}"), L.get("random50")
            if van and vp and rp and r:
                base = med_dlp(van, vp)
                thr = max(0.05, 3 * base)
                thr_used.append(thr)
                got = med_dlp(r, rp)
                rec(m, f"T3 random50 vs random50_{pert} (median dlp)", got <= thr,
                    f"{got:.4f} <= threshold {thr:.4f} (vanilla pair {base:.4f})")
        if L.get("random50_shift7") and L.get("random50_nopc") and thr_used:
            thr = max(thr_used)
            got = med_dlp(L["random50_nopc"], L["random50_shift7"])
            rec(m, "T3n negative control: deliberate position shift detected", got > thr,
                f"median dlp {got:.4f} vs threshold {thr:.4f}")
        ev, ka = L.get("vanilla_evs_nopc"), L.get("keep_all_nopc")
        if ev and ka:
            mx = max(dlp(ev[k], ka[k]) for k in ev)
            mxv = max(dlp(ev[k], vn[k]) for k in ev) if vn else float("nan")
            rec(m, "D1 keep_all_nopc vs vanilla_evs_nopc (diagnostic)", mx == 0.0,
                f"max dlp {mx:.2e}; vanilla_evs_nopc vs vanilla_nopc {mxv:.2e}", hard=False)
        tr = L.get("random50_transpose")
        if tr and L.get("random50_nopc") and thr_used:
            got = med_dlp(L["random50_nopc"], tr)
            rec(m, "D2 image-position transpose detected by T3 statistic (diagnostic)", got > max(thr_used),
                f"median dlp {got:.4f} vs threshold {max(thr_used):.4f}", hard=False)
        for name in ("vanilla", "random50", "divprune25", "pool2x2"):
            run = L.get(name)
            if run:
                p = sum(run[(0, q)]["output"] == run[(1, q)]["output"] for (r, q) in run if r == 0)
                n = sum(1 for (r, q) in run if r == 0)
                rec(m, f"second pass (prefix cache) {name}", p == n, f"pass0 == pass1 on {p}/{n}", hard=False)
    print("| model | check | result | detail |\n|---|---|---|---|")
    print("\n".join(rows))
    print("ALL_HARD_CHECKS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
