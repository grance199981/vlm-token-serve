"""Follow-up to condition_analysis.py (post hoc, hypothesis-generating only; see
docs/W3_CONDITION_ANALYSIS.md §5): is the pruning advantage tied to how *local* the
answer evidence is?

  TextVQA: bin by the number of OCR tokens in the image (few tokens = local text).
  DocVQA : group by the dataset's question_types labels (multi-label; a sample counts
           in every type it carries).
d = score(downsample) - score(RATE), paired by doc_id; negative = RATE better.
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

from condition_analysis import KEEPS, MODELS, boot_mean, find_samples, load_samples, quartile_bins


def rows(root, pat, limit, cols):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "data/lmms-lab", pat))):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=cols):
            for r in batch.to_pylist():
                if len(out) >= limit:
                    return out
                out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    groups = {}
    tv = rows(a.root, "textvqa/data/validation-*.parquet", 1500, ["question", "ocr_tokens"])
    n_ocr = [len(r["ocr_tokens"] or []) for r in tv]
    b, q = quartile_bins(n_ocr)
    groups["vts_textvqa_val"] = ("exact_match", {i: [f"ocr_Q{b[i] + 1}"] for i in range(len(tv))},
                                 {"ocr_token_quartiles": q})
    dv = rows(a.root, "DocVQA/DocVQA/validation-*.parquet", 1000, ["question", "question_types"])
    groups["vts_docvqa_val"] = ("anls", {i: list(r["question_types"] or ["none"]) for i, r in enumerate(dv)}, {})
    out = []
    for task, (metric, labels, info) in groups.items():
        print(task, info)
        for model in MODELS:
            b0 = load_samples(find_samples(a.root, model, "B0", task), metric)
            for keep, kname in KEEPS.items():
                if keep == "50":
                    continue
                ds = find_samples(a.root, model, f"downsample_{kname}", task)
                rt = find_samples(a.root, model, f"rate_{kname}", task)
                if not ds or not rt:
                    continue
                ds, rt = load_samples(ds, metric), load_samples(rt, metric)
                names = sorted({g for v in labels.values() for g in v})
                for g in names:
                    ids = [i for i, v in labels.items() if g in v and i in ds and i in rt and i in b0]
                    if len(ids) < 30:
                        continue
                    d = [ds[i][0] - rt[i][0] for i in ids]
                    m, lo, hi = boot_mean(d)
                    row = {"task": task, "model": model, "keep": keep, "group": g, "n": len(ids), "d": [m, lo, hi],
                           "delta_downsample": float(np.mean([ds[i][0] - b0[i][0] for i in ids])),
                           "delta_rate": float(np.mean([rt[i][0] - b0[i][0] for i in ids])),
                           "b0": float(np.mean([b0[i][0] for i in ids]))}
                    out.append(row)
                    print(f"{task:16s} {model[:8]:8s} keep={keep:4s} {g:16s} n={len(ids):4d} b0={row['b0']:.3f} "
                          f"d={m:+.3f} [{lo:+.3f},{hi:+.3f}] ds={row['delta_downsample']:+.3f} rate={row['delta_rate']:+.3f}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
