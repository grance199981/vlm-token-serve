"""Route B (docs/W3_CONDITION_ANALYSIS.md §9): is the pruning advantage on TextVQA about
few words (n_words) or small text (h_px)? CPU only.

d = score(downsample) - score(RATE) per question, TextVQA val rows 0-4999 (0-1499 from W3,
1500-4999 from the hold-out run A). Regression d ~ z(log h_px) + z(log1p n_words) per
model x keep, bootstrap CIs; decision rule fixed in §9.3.

Usage: python route_b.py --root $VTS_ROOT --holdout-run <run dir> --out <json>
"""
import argparse
import collections
import glob
import json
import os
import random
import re

import numpy as np
import pyarrow.parquet as pq

from condition_analysis import MODELS, find_samples, load_samples

KEEPS = {"25": "k25", "12.5": "k125"}
HOLDOUT_START = 1500


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def word_key(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())


def load_questions(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "data/lmms-lab/textvqa/data/validation-*.parquet"))):
        rows += pq.read_table(f, columns=["question", "image_id", "image_width", "image_height",
                                          "ocr_tokens", "answers"]).to_pylist()
    assert len(rows) == 5000
    ocr = {r["image_id"]: r for r in json.load(open(os.path.join(root, "data/textvqa_ocr/TextVQA_Rosetta_OCR_v0.2_val.json")))["data"]}
    out, same_tokens = [], 0
    for r in rows:
        o = ocr[r["image_id"]]
        info = o["ocr_info"]
        same_tokens += list(r["ocr_tokens"] or []) == list(o["ocr_tokens"] or [])
        H, W = r["image_height"], r["image_width"]
        heights = [b["bounding_box"]["height"] * H for b in info]
        area = min(1.0, sum(b["bounding_box"]["width"] * b["bounding_box"]["height"] for b in info))
        ans = collections.Counter(a.strip().lower() for a in r["answers"]).most_common(1)[0][0]
        ans_words = {word_key(w) for w in ans.split()} - {""}
        matched = [b["bounding_box"]["height"] * H for b in info if word_key(b["word"]) in ans_words]
        out.append({"question": r["question"], "n_words": len(info),
                    "h_px": float(np.median(heights)) if heights else float("nan"),
                    "area": area, "h_ans_px": float(np.median(matched)) if matched else float("nan")})
    return out, same_tokens


def scores(root, holdout_run, model, cfg):
    s = {}
    early = load_samples(find_samples(root, model, cfg, "vts_textvqa_val"), "exact_match")
    for i, v in early.items():
        if i < HOLDOUT_START:
            s[i] = v
    path = sorted(glob.glob(os.path.join(holdout_run, model, cfg, "lmms_eval", "**",
                                         "*samples_vts_textvqa_holdout.jsonl"), recursive=True))[-1]
    for i, v in load_samples(path, "exact_match").items():
        s[i + HOLDOUT_START] = v
    return s


def ols(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0][1:]


def fit(xcols, y, n_boot=1000, seed=0):
    X = np.column_stack([(c - c.mean()) / c.std() for c in xcols])
    coef = ols(X, y)
    rng = random.Random(seed)
    idx = list(range(len(y)))
    boots = []
    for _ in range(n_boot):
        b = [rng.choice(idx) for _ in idx]
        boots.append(ols(X[b], y[b]))
    boots = np.sort(np.array(boots), axis=0)
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return [(float(c), float(l), float(h)) for c, l, h in zip(coef, lo, hi)]


def classify(c_h, c_n):
    sig_h, sig_n = c_h[1] > 0, c_n[1] > 0  # positive and CI above 0
    if sig_h and not sig_n:
        return "small-text"
    if sig_n and not sig_h:
        return "few-words"
    if sig_h and sig_n:
        return "both"
    return "neither"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--holdout-run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qs, same = load_questions(a.root)
    res = {"ocr_tokens_identical_to_parquet": same, "n": len(qs),
           "n_no_ocr": sum(q["n_words"] == 0 for q in qs), "combos": []}
    has = [i for i, q in enumerate(qs) if q["n_words"] > 0]
    lh = np.log(np.array([qs[i]["h_px"] for i in has]))
    ln = np.log1p(np.array([qs[i]["n_words"] for i in has], dtype=float))
    res["corr_logh_logn"] = float(np.corrcoef(lh, ln)[0, 1])
    print(f"OCR tokens identical to parquet: {same}/5000; images without OCR words: {res['n_no_ocr']}; "
          f"corr(log h_px, log1p n_words) = {res['corr_logh_logn']:+.3f}")
    print(f"h_px quartiles: {np.quantile(np.exp(lh), [0.25, 0.5, 0.75]).round(1).tolist()}")
    verdicts = []
    for model in MODELS:
        for keep, kname in KEEPS.items():
            ds = scores(a.root, a.holdout_run, model, f"downsample_{kname}")
            rt = scores(a.root, a.holdout_run, model, f"rate_{kname}")
            ids = [i for i in has if i in ds and i in rt]
            mism = sum(not norm(rt[i][2]).startswith(norm(qs[i]["question"])[:40]) for i in ids)
            if mism > 0.005 * len(ids):
                raise SystemExit(f"alignment failed {model} {keep}: {mism}")
            y = np.array([ds[i][0] - rt[i][0] for i in ids])
            h = np.log(np.array([qs[i]["h_px"] for i in ids]))
            n = np.log1p(np.array([qs[i]["n_words"] for i in ids], dtype=float))
            ar = np.log(np.array([max(qs[i]["area"], 1e-6) for i in ids]))
            c_h, c_n = fit([h, n], y)
            verdict = classify(c_h, c_n)
            verdicts.append(verdict)
            c_h2, c_a = fit([h, ar], y)
            sub = [k for k, i in enumerate(ids) if not np.isnan(qs[i]["h_ans_px"])]
            ha = np.log(np.array([qs[ids[k]]["h_ans_px"] for k in sub]))
            c_ha, c_n2 = fit([ha, n[sub]], y[sub])
            mh, mn = np.median(h), np.median(n)
            grid = {}
            for hn, hm in (("small", h <= mh), ("large", h > mh)):
                for nn, nm in (("few", n <= mn), ("many", n > mn)):
                    sel = hm & nm
                    grid[f"{hn}_text/{nn}_words"] = (float(y[sel].mean()), int(sel.sum()))
            no_ocr = [i for i, q in enumerate(qs) if q["n_words"] == 0 and i in ds and i in rt]
            row = {"model": model, "keep": keep, "n": len(ids), "coef_h": c_h, "coef_n": c_n, "verdict": verdict,
                   "alt_area": {"coef_h": c_h2, "coef_area": c_a},
                   "alt_answer_height": {"n": len(sub), "coef_h_ans": c_ha, "coef_n": c_n2},
                   "grid_mean_d": grid,
                   "no_ocr_mean_d": (float(np.mean([ds[i][0] - rt[i][0] for i in no_ocr])), len(no_ocr))}
            res["combos"].append(row)
            f = lambda c: f"{c[0]:+.4f}[{c[1]:+.4f},{c[2]:+.4f}]"  # noqa: E731
            print(f"{model[:8]:8s} keep={keep:4s} n={len(ids)} coef_h={f(c_h)} coef_n={f(c_n)} -> {verdict}")
            print(f"   alt area: coef_h={f(c_h2)} coef_area={f(c_a)}; answer-height subset n={len(sub)}: "
                  f"coef_h_ans={f(c_ha)} coef_n={f(c_n2)}")
            print("   2x2 mean d (n): " + "; ".join(f"{k}={v[0]:+.3f}({v[1]})" for k, v in grid.items())
                  + f"; no-OCR images d={row['no_ocr_mean_d'][0]:+.3f}({row['no_ocr_mean_d'][1]})")
    top = collections.Counter(verdicts).most_common(1)[0]
    res["verdict"] = top[0] if top[1] >= 3 else "inconsistent"
    print("verdicts", verdicts, "->", res["verdict"])
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
