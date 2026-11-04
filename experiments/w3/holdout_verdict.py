"""H-C4 hold-out verdict with the criteria locked in docs/W3_CONDITION_ANALYSIS.md §7.

For each model x keep: (i) d in the OCR <= 4 group has CI upper < 0; (ii) d(>16) - d(<=4) has
CI lower > 0, with d = score(downsample) - score(RATE) per question (exact match).
>= 3 of 4 combos with (i) and (ii): confirmed; 1-2: partial; 0: not supported.
B0 (W1, all 5000 TextVQA val rows) is used for the reported deltas only (doc_id + 1500).

Usage: python holdout_verdict.py --root $VTS_ROOT --run <holdout run dir> --out <json>
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq

from condition_analysis import B0_RUN, MODELS, boot_diff, boot_mean, load_samples

START = 1500
KEEPS = {"25": "k25", "12.5": "k125"}


def group(n_ocr):
    return "le4" if n_ocr <= 4 else ("gt16" if n_ocr > 16 else "mid")


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = pq.read_table(glob.glob(os.path.join(a.root, "data/holdout/textvqa_val_*.parquet"))[0],
                      columns=["question", "ocr_tokens"]).to_pylist()
    grp = {i: group(len(r["ocr_tokens"] or [])) for i, r in enumerate(t)}
    res = {"group_sizes": {g: sum(v == g for v in grp.values()) for g in ("le4", "mid", "gt16")}, "combos": []}
    passed = 0
    for model in MODELS:
        b0_all = load_samples(sorted(glob.glob(os.path.join(a.root, "runs", B0_RUN, "lmms_eval", model,
                                                            "*samples_vts_textvqa_val.jsonl")))[-1], "exact_match")
        for keep, kname in KEEPS.items():
            sc = {}
            for m in ("downsample", "rate"):
                d = os.path.join(a.run, model, f"{m}_{kname}")
                if open(os.path.join(d, "valid.txt")).read().strip() != "yes":
                    raise SystemExit(f"invalid config {d}")
                sc[m] = load_samples(sorted(glob.glob(os.path.join(d, "lmms_eval", "**", "*samples_vts_textvqa_holdout.jsonl"),
                                                      recursive=True))[-1], "exact_match")
            ids = sorted(set(sc["downsample"]) & set(sc["rate"]))
            mism = sum(not norm(sc["rate"][i][2]).startswith(norm(t[i]["question"])[:40]) for i in ids)
            mism_b0 = sum(not norm(b0_all[i + START][2]).startswith(norm(t[i]["question"])[:40]) for i in ids)
            if mism > 0.005 * len(ids) or mism_b0 > 0.005 * len(ids):
                raise SystemExit(f"alignment failed {model} {keep}: {mism} / {mism_b0}")
            d = {g: [sc["downsample"][i][0] - sc["rate"][i][0] for i in ids if grp[i] == g] for g in ("le4", "mid", "gt16")}
            row = {"model": model, "keep": keep, "n": len(ids),
                   "d": {g: boot_mean(v) for g, v in d.items()},
                   "trend_gt16_minus_le4": boot_diff(d["gt16"], d["le4"]),
                   "delta_vs_b0": {m: {g: float(np.mean([sc[m][i][0] - b0_all[i + START][0] for i in ids if grp[i] == g]))
                                       for g in ("le4", "mid", "gt16")} for m in sc}}
            row["crit_i"] = row["d"]["le4"][2] < 0
            row["crit_ii"] = row["trend_gt16_minus_le4"][1] > 0
            row["pass"] = row["crit_i"] and row["crit_ii"]
            passed += row["pass"]
            res["combos"].append(row)
            print(f"{model[:8]:8s} keep={keep:4s} n={len(ids)} "
                  + " ".join(f"d_{g}={v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]" for g, v in row["d"].items())
                  + f" trend={row['trend_gt16_minus_le4'][0]:+.3f}[{row['trend_gt16_minus_le4'][1]:+.3f},"
                  f"{row['trend_gt16_minus_le4'][2]:+.3f}] (i)={row['crit_i']} (ii)={row['crit_ii']}")
    res["passed"] = passed
    res["verdict"] = "confirmed" if passed >= 3 else ("partial" if passed >= 1 else "not supported")
    print("group sizes", res["group_sizes"], "| combos passed", passed, "of 4 ->", res["verdict"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
