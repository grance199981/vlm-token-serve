"""Route C verdicts with the criteria locked in docs/W3_CONDITION_ANALYSIS.md §12.

d = ANLS(downsample) - ANLS(RATE) per question (negative = RATE better).
ST-VQA (main): groups by Rosetta word count (<=4, 5-16, >16); (i) d(<=4) CI upper < 0,
  (ii) d(>16) - d(<=4) CI lower > 0; >= 3 of 4 combos -> replicated. Secondary: d ~ z(log area).
InfoVQA (dense end): C-D1 overall d CI lower > 0 in >= 3 combos; C-D2 d ~ z(log area) coef CI lower > 0.

Usage: python routec_verdict.py --root $VTS_ROOT --run <run dir> --out <json>
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "w3"))
from condition_analysis import boot_diff, boot_mean  # noqa: E402

MODELS = ["Qwen3-VL-4B-Instruct", "Qwen3.5-4B"]
KEEPS = {"25": "k25", "12.5": "k125"}
TASKS = {"vts_stvqa_subval": "stvqa_features.json", "vts_infovqa_val": "infovqa_features.json"}


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def load(run, model, cfg, task):
    d = os.path.join(run, model, cfg)
    if open(os.path.join(d, "valid.txt")).read().strip() != "yes":
        raise SystemExit(f"invalid config {d}")
    f = sorted(glob.glob(os.path.join(d, "lmms_eval", "**", f"*samples_{task}.jsonl"), recursive=True))[-1]
    out = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        out[int(r["doc_id"])] = (float(r["anls"]), r.get("input", ""))
    return out


def slope(x, y, n_boot=1000, seed=0):
    z = (x - x.mean()) / x.std()
    A = np.column_stack([np.ones(len(y)), z])
    coef = np.linalg.lstsq(A, y, rcond=None)[0][1]
    rng = random.Random(seed)
    idx = list(range(len(y)))
    bs = sorted(np.linalg.lstsq(A[b], y[b], rcond=None)[0][1] for b in ([rng.choice(idx) for _ in idx] for _ in range(n_boot)))
    return float(coef), float(bs[int(0.025 * n_boot)]), float(bs[int(0.975 * n_boot) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for task, ffile in TASKS.items():
        feats = {int(k): v for k, v in json.load(open(os.path.join(a.root, "data/routec", ffile))).items()}
        rows, b0s = [], {}
        for model in MODELS:
            b0 = load(a.run, model, "b0", task)
            b0s[model] = float(np.mean([v[0] for v in b0.values()]))
            for keep, k in KEEPS.items():
                ds, rt = load(a.run, model, f"downsample_{k}", task), load(a.run, model, f"rate_{k}", task)
                ids = sorted(set(ds) & set(rt) & set(b0) & set(feats))
                mism = sum(not norm(rt[i][1]).startswith(norm(feats[i]["question"])[:40]) for i in ids)
                if mism > 0.005 * len(ids):
                    raise SystemExit(f"alignment failed {task} {model} {keep}: {mism}/{len(ids)}")
                d = {i: ds[i][0] - rt[i][0] for i in ids}
                row = {"model": model, "keep": keep, "n": len(ids), "alignment_mismatch": mism,
                       "overall_d": boot_mean(list(d.values())),
                       "delta_vs_b0": {m: float(np.mean([s[i][0] - b0[i][0] for i in ids])) for m, s in (("downsample", ds), ("rate", rt))}}
                if task == "vts_stvqa_subval":
                    g = {"le4": [d[i] for i in ids if feats[i]["n_words"] <= 4],
                         "mid": [d[i] for i in ids if 4 < feats[i]["n_words"] <= 16],
                         "gt16": [d[i] for i in ids if feats[i]["n_words"] > 16]}
                    row["groups"] = {k2: (boot_mean(v), len(v)) for k2, v in g.items()}
                    row["trend"] = boot_diff(g["gt16"], g["le4"])
                    row["crit_i"] = row["groups"]["le4"][0][2] < 0
                    row["crit_ii"] = row["trend"][1] > 0
                    row["pass"] = row["crit_i"] and row["crit_ii"]
                    sel = [i for i in ids if feats[i]["n_words"] > 0]
                else:
                    row["pass_D1"] = row["overall_d"][1] > 0
                    sel = [i for i in ids if feats[i]["n_words"] > 0]
                x = np.log(np.array([max(feats[i]["area"], 1e-6) for i in sel]))
                row["area_slope"] = slope(x, np.array([d[i] for i in sel]))
                row["area_pass"] = row["area_slope"][1] > 0
                rows.append(row)
                print(f"{task:17s} {model[:8]:8s} keep={keep:4s} n={len(ids)} overall d={row['overall_d'][0]:+.3f}"
                      f"[{row['overall_d'][1]:+.3f},{row['overall_d'][2]:+.3f}] "
                      + (" ".join(f"{k2}={v[0][0]:+.3f}[{v[0][1]:+.3f},{v[0][2]:+.3f}](n={v[1]})" for k2, v in row["groups"].items())
                         + f" trend={row['trend'][0]:+.3f}[{row['trend'][1]:+.3f},{row['trend'][2]:+.3f}] pass={row['pass']}"
                         if "groups" in row else f"D1={row['pass_D1']}")
                      + f" area_slope={row['area_slope'][0]:+.4f}[{row['area_slope'][1]:+.4f},{row['area_slope'][2]:+.4f}]"
                      + f" ds={row['delta_vs_b0']['downsample']:+.3f} rate={row['delta_vs_b0']['rate']:+.3f}")
        out = {"b0_anls": b0s, "combos": rows}
        if task == "vts_stvqa_subval":
            n = sum(r["pass"] for r in rows)
            out["verdict"] = "replicated" if n >= 3 else ("partial" if n >= 1 else "not replicated")
        else:
            n = sum(r["pass_D1"] for r in rows)
            out["verdict_D1"] = "supported" if n >= 3 else "not supported"
        out["area_verdict"] = "supported" if sum(r["area_pass"] for r in rows) >= 3 else "not supported"
        res[task] = out
        print(task, "B0 ANLS", {k: round(v, 3) for k, v in b0s.items()},
              {k: v for k, v in out.items() if "verdict" in k})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
