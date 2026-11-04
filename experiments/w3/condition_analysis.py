"""W3 condition analysis (docs/W3_CONDITION_ANALYSIS.md): when does pruning beat downsampling?

Per sample: d = score(downsample) - score(RATE), paired by doc_id, binned within each task
by the B0 input-token count (quartiles) and, for DocVQA / ChartQA, by ink fraction.
CPU only. The run directories under $VTS_ROOT/runs are passed by environment variable
(they are also imported by holdout_verdict.py, route_b.py and condition_analysis_extra.py):
  VTS_W1_ACC_RUN=<W1 accuracy run> VTS_W3_RUNS="<W3 run> [<W3 run> ...]" \
      python experiments/w3/condition_analysis.py --root $VTS_ROOT --out <json>
"""
import argparse
import glob
import io
import json
import os
import random
import re

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

TASKS = {  # task -> (parquet glob, limit, metric key)
    "vts_textvqa_val": ("textvqa/data/validation-*.parquet", 1500, "exact_match"),
    "vts_docvqa_val": ("DocVQA/DocVQA/validation-*.parquet", 1000, "anls"),
    "vts_chartqa": ("ChartQA/data/test-*.parquet", 2500, "relaxed_overall"),
}
B0_RUN = os.environ.get("VTS_W1_ACC_RUN", "")  # W1 accuracy run (uncompressed B0)
RUNS = os.environ.get("VTS_W3_RUNS", "").split()  # W3 sweep runs
MODELS = ["Qwen3-VL-4B-Instruct", "Qwen3.5-4B"]
KEEPS = {"50": "k50", "25": "k25", "12.5": "k125"}
METHODS = ["downsample", "rate", "divprune", "random"]
UNCHECKED_OK = {("Qwen3-VL-4B-Instruct", "random_k25_s0")}  # valid run that predates valid.txt


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_images(root, task):
    pat, limit, _ = TASKS[task]
    out = []
    for f in sorted(glob.glob(os.path.join(root, "data/lmms-lab", pat))):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if len(out) >= limit:
                    return out
                im = Image.open(io.BytesIO(row["image"]["bytes"]))
                w, h = im.size
                ink = None
                if task != "vts_textvqa_val":
                    g = im.convert("L")
                    g.thumbnail((512, 512))
                    a = np.asarray(g)
                    ink = float((a < 200).mean())
                out.append({"question": row["question"], "w": w, "h": h, "ink": ink})
    return out


def load_samples(path, metric):
    res = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        tok = (r.get("token_counts") or [{}])[0].get("input_tokens")
        res[int(r["doc_id"])] = (float(r[metric]), tok, r.get("input", ""))
    return res


def find_samples(root, model, cfg, task):
    if cfg == "B0":
        pats = [os.path.join(root, "runs", B0_RUN, "lmms_eval", model, f"*samples_{task}.jsonl")]
    else:
        pats = []
        for run in RUNS:
            d = os.path.join(root, "runs", run, model, cfg)
            vf = os.path.join(d, "valid.txt")
            ok = (open(vf).read().strip() == "yes") if os.path.exists(vf) else (model, cfg) in UNCHECKED_OK
            if ok:
                pats.append(os.path.join(d, "lmms_eval", "**", f"*samples_{task}.jsonl"))
    for p in pats:
        files = sorted(glob.glob(p, recursive=True))
        if files:
            return files[-1]
    return None


def boot_mean(x, n=1000, seed=0):
    rng = random.Random(seed)
    k = len(x)
    if k == 0:
        return (float("nan"),) * 3
    means = sorted(sum(rng.choice(x) for _ in range(k)) / k for _ in range(n))
    return sum(x) / k, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def boot_diff(x_hi, x_lo, n=1000, seed=0):
    rng = random.Random(seed)
    diffs = sorted(sum(rng.choice(x_hi) for _ in x_hi) / len(x_hi) - sum(rng.choice(x_lo) for _ in x_lo) / len(x_lo)
                   for _ in range(n))
    return sum(x_hi) / len(x_hi) - sum(x_lo) / len(x_lo), diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def quartile_bins(values):
    qs = np.quantile(values, [0.25, 0.5, 0.75])
    return [int(np.searchsorted(qs, v, side="right")) for v in values], [float(q) for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {"tasks": {}, "rows": []}
    for task, (_, limit, metric) in TASKS.items():
        imgs = load_images(a.root, task)
        tinfo = {"n_images": len(imgs)}
        for model in MODELS:
            b0_path = find_samples(a.root, model, "B0", task)
            b0 = load_samples(b0_path, metric)
            # alignment check: sample input starts with the dataset question
            ids = sorted(i for i in b0 if i < len(imgs))
            mism = sum(not norm(b0[i][2]).startswith(norm(imgs[i]["question"])[:40]) for i in ids)
            tinfo[f"{model}_alignment_mismatch"] = mism
            if mism > 0.005 * len(ids):
                raise SystemExit(f"alignment failed for {task} {model}: {mism}/{len(ids)}")
            toks = [b0[i][1] for i in ids]
            tinfo[f"{model}_b0_tokens_median"] = float(np.median(toks))
            tinfo[f"{model}_b0_tokens_q"] = [float(q) for q in np.quantile(toks, [0.25, 0.5, 0.75])]
            size_bin, size_q = quartile_bins(toks)
            bins = {"size": dict(zip(ids, size_bin))}
            if imgs[0]["ink"] is not None:
                ink_bin, _ = quartile_bins([imgs[i]["ink"] for i in ids])
                bins["ink"] = dict(zip(ids, ink_bin))
            for keep, kname in KEEPS.items():
                cfgs = {m: (f"{m}_{kname}_s0" if (m == "random" and kname == "k25") else f"{m}_{kname}") for m in METHODS}
                scores = {}
                for m, cfg in cfgs.items():
                    p = find_samples(a.root, model, cfg, task)
                    if p:
                        scores[m] = load_samples(p, metric)
                if "downsample" not in scores or "rate" not in scores:
                    continue
                common = [i for i in ids if i in scores["downsample"] and i in scores["rate"]]
                for var, bmap in bins.items():
                    per_bin = []
                    for q in range(4):
                        sel = [i for i in common if bmap[i] == q]
                        d = [scores["downsample"][i][0] - scores["rate"][i][0] for i in sel]
                        row = {"task": task, "model": model, "keep": keep, "var": var, "bin": q + 1, "n": len(sel),
                               "d": boot_mean(d)}
                        for m in scores:
                            row[f"delta_{m}"] = float(np.mean([scores[m][i][0] - b0[i][0] for i in sel if i in scores[m]]))
                        per_bin.append((row, d))
                        report["rows"].append(row)
                    trend = boot_diff(per_bin[3][1], per_bin[0][1])
                    report["rows"].append({"task": task, "model": model, "keep": keep, "var": var, "bin": "Q4-Q1",
                                           "n": len(common), "d": trend})
        report["tasks"][task] = tinfo
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report["tasks"], indent=1))
    for r in report["rows"]:
        m, lo, hi = r["d"]
        extra = " ".join(f"{k[6:]}={v:+.3f}" for k, v in r.items() if k.startswith("delta_"))
        print(f"{r['task']:17s} {r['model'][:8]:8s} keep={r['keep']:4s} {r['var']:4s} {str(r['bin']):5s} n={r['n']:4d} "
              f"d={m:+.3f} [{lo:+.3f},{hi:+.3f}] {extra}")


if __name__ == "__main__":
    main()
