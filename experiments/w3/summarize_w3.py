"""Summarize the W3 sweep against the W1 B0 baseline (paired by doc_id).

Usage: python summarize_w3.py --b0 <W1 run dir> --runs <W3 run dir> [<W3 run dir> ...] [--csv out.csv]

Per task score per sample:
  chartqa relaxed_overall, textvqa exact_match, docvqa anls,
  mmbench: first option letter of the prediction == answer (simple static rule,
           applied identically to B0 and every config; the lmms-eval aggregate is
           also reported)
  mme: aggregate perception / cognition from results.json (no per-sample CI)
Delta = config - B0 on the same doc_ids, paired bootstrap 95% CI (2000, seed 0).
Only configs whose valid.txt is "yes" are used; runs that predate valid.txt need
--include-unchecked model/config (checked by hand).
"""
import argparse
import csv
import glob
import json
import os
import random
import re

TASKS = {"vts_chartqa": "relaxed_overall", "vts_textvqa_val": "exact_match", "vts_docvqa_val": "anls",
         "vts_mmbench_en_dev_static": None}


def mmbench_correct(rec):
    g = rec["gpt_eval_score"]
    m = re.match(r"\s*\(?([A-E])\b", g["prediction"])
    return float(bool(m) and m.group(1) == g["answer"])


def load_samples(model_dir, task):
    files = sorted(glob.glob(os.path.join(model_dir, "**", f"*_samples_{task}.jsonl"), recursive=True))
    if not files:
        return {}
    out = {}
    for line in open(files[-1], encoding="utf-8"):
        r = json.loads(line)
        score = mmbench_correct(r) if TASKS[task] is None else float(r[TASKS[task]])
        tok = (r.get("token_counts") or [{}])[0].get("input_tokens")
        out[int(r["doc_id"])] = (score, tok)
    return out


def load_mme(model_dir):
    vals = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "**", "*_results.json"), recursive=True)):
        res = json.load(open(f, encoding="utf-8")).get("results", {})
        if "vts_mme" in res:
            for k, v in res["vts_mme"].items():
                if k.startswith("mme_") and "stderr" not in k:
                    vals[k.split(",")[0]] = v
    return vals


def boot(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--include-unchecked", nargs="*", default=[],
                    help="model/config without valid.txt that were verified by hand")
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        for model_dir in sorted(glob.glob(os.path.join(run, "*"))):
            if not os.path.isdir(model_dir):
                continue
            model = os.path.basename(model_dir)
            b0_dir = os.path.join(a.b0, "lmms_eval", model)
            for cfg_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
                cfg = os.path.basename(cfg_dir)
                vf = os.path.join(cfg_dir, "valid.txt")
                ok = (open(vf).read().strip() == "yes" if os.path.exists(vf)
                      else f"{model}/{cfg}" in a.include_unchecked)
                if not ok:
                    print(f"skip {run}/{model}/{cfg} (not valid)")
                    continue
                for task in TASKS:
                    b0, cur = load_samples(b0_dir, task), load_samples(cfg_dir, task)
                    ids = sorted(set(b0) & set(cur))
                    if not ids:
                        continue
                    d = [cur[i][0] - b0[i][0] for i in ids]
                    lo, hi = boot(d)
                    tb = [b0[i][1] for i in ids if b0[i][1]]
                    tc = [cur[i][1] for i in ids if cur[i][1]]
                    rows.append({"model": model, "config": cfg, "task": task, "n": len(ids),
                                 "b0": 100 * sum(b0[i][0] for i in ids) / len(ids),
                                 "score": 100 * sum(cur[i][0] for i in ids) / len(ids),
                                 "delta": 100 * sum(d) / len(d), "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                                 "tok_ratio": (sum(tc) / len(tc)) / (sum(tb) / len(tb)) if tb and tc else None})
                mme_b0, mme = load_mme(b0_dir), load_mme(cfg_dir)
                for k in mme:
                    if k in mme_b0:
                        rows.append({"model": model, "config": cfg, "task": f"vts_mme:{k}", "n": 2374,
                                     "b0": mme_b0[k], "score": mme[k], "delta": mme[k] - mme_b0[k],
                                     "ci_lo": None, "ci_hi": None, "tok_ratio": None})
    print("| model | config | task | n | B0 | score | delta (95% CI) | input-token ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        ci = f" [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]" if r["ci_lo"] is not None else ""
        tr = f"{r['tok_ratio']:.2f}" if r["tok_ratio"] else "—"
        print(f"| {r['model']} | {r['config']} | {r['task']} | {r['n']} | {r['b0']:.1f} | {r['score']:.1f} | "
              f"{r['delta']:+.1f}{ci} | {tr} |")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
