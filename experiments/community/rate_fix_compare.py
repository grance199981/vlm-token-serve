"""Does the RATE PR #60963 fix (commit 90dc2a751f: ascending sort, guard `denom == 0`) select the
same tokens as this plugin's fix (descending sort, guard `denom <= 0`)?

Reading the code: with an ascending sort the denominator is negative, so q_scale < 0 and
round(s * q_scale) = -round(s * |q_scale|): the buckets are the same as with a descending sort,
but each bucket's representative (the pivot) is its LOWEST-score token instead of its highest,
and the first pivot is the global minimum instead of the maximum. So the pivot sets, and hence
the selections, can differ. This script measures by how much, on synthetic inputs, with:
  A. the same projection matrix in both (isolates the sort-direction choice);
  B. each implementation's own projections (what users would actually get);
plus controls: identical function vs itself (overlap must be 1), the PR before the fix (must keep
a prefix), and a random selection of the same size (chance-level overlap).

Usage: python rate_fix_compare.py --pr-file <rate.py at the PR head> --plugin-src <repo>/src
"""
import argparse
import importlib.util
import json
import sys

import torch
import torch.nn.functional as F


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs(kind, n, d, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "gauss":
        return torch.randn(n, d, generator=g)
    # clustered: 12 centres with unequal sizes plus noise, closer to image features than iid noise
    c = torch.randn(12, d, generator=g)
    w = torch.rand(12, generator=g) ** 2 + 0.05
    lab = torch.multinomial(w / w.sum(), n, replacement=True, generator=g)
    return c[lab] + 0.6 * torch.randn(n, d, generator=g)


def overlap(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-file", required=True)
    ap.add_argument("--plugin-src", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    pr = load(a.pr_file, "rate_pr_fixed")
    sys.path.insert(0, a.plugin_src)
    from vts_plugin import select as ours  # noqa: E402

    rows = []
    for kind in ("gauss", "clustered"):
        for n in (196, 784, 3136):
            for d in (2560,):
                for q in (0.5, 0.75):
                    for s in range(5):
                        x = inputs(kind, n, d, 1000 * s + n).to(torch.bfloat16)
                        k = pr.compute_retained_tokens_count(n, q)
                        # our implementation, our projections (seed 0 as in all W3/W4 runs)
                        mine = ours.rate_indices(x, k, seed=0)
                        # PR fixed, its own projections
                        pr._get_random_projections.cache_clear()
                        theirs_own = pr.compute_retention_mask_rate(x, q).nonzero().flatten()
                        # PR fixed with OUR projection matrix (isolates the sort direction)
                        g = torch.Generator(device="cpu").manual_seed(0)
                        P = F.normalize(torch.randn(8, d, generator=g).to(x.dtype), p=2, dim=-1)
                        orig = pr._get_random_projections
                        pr._get_random_projections = lambda **kw: P
                        theirs_sameproj = pr.compute_retention_mask_rate(x, q).nonzero().flatten()
                        pr._get_random_projections = orig
                        # our code with ascending sort and the relaxed guard == PR fix? (same projections)
                        sc = ours._rate_scores(x, 32, max(1, int(0.01 * n)), 8, 0, descending=False)
                        # ours guards denom <= 0, so with ascending it returns zeros: emulate `== 0`
                        mine_desc_scores = ours._rate_scores(x, 32, max(1, int(0.01 * n)), 8, 0, descending=True)
                        rnd = torch.randperm(n, generator=torch.Generator().manual_seed(s))[:k]
                        rows.append({
                            "kind": kind, "n": n, "q": q, "seed": s, "k": k,
                            "self": overlap(mine, ours.rate_indices(x, k, seed=0)),
                            "same_proj": overlap(mine, theirs_sameproj),
                            "own_proj": overlap(mine, theirs_own),
                            "random": overlap(mine, rnd),
                            "prefix_check_theirs": bool(torch.equal(theirs_own.sort().values, torch.arange(k))),
                            "ours_ascending_is_constant": bool(sc.std() == 0),
                            "ours_desc_has_spread": bool(mine_desc_scores.float().std() > 0),
                        })
    # negative control: PR before the fix (guard `denom <= 0`) must keep a prefix
    pre = open(a.pr_file).read().replace("if denom == 0:", "if denom <= 0:")
    assert pre != open(a.pr_file).read(), "fixed guard not found in PR file"
    tmp = "/tmp/rate_pr_prefix_ctrl.py"
    open(tmp, "w").write(pre)
    old = load(tmp, "rate_pr_old")
    x = inputs("gauss", 784, 2560, 7).to(torch.bfloat16)
    m = old.compute_retention_mask_rate(x, 0.5)
    prefix_ctrl = bool(m[: int(m.sum())].all())

    def agg(key, sel):
        v = [r[key] for r in rows if sel(r)]
        return round(sum(v) / len(v), 3), round(min(v), 3), round(max(v), 3)

    out = {"n_cases": len(rows), "control_pre_fix_keeps_prefix": prefix_ctrl,
           "fixed_PR_never_prefix": not any(r["prefix_check_theirs"] for r in rows),
           "ours_desc_scores_have_spread": all(r["ours_desc_has_spread"] for r in rows),
           "summary": {}}
    for kind in ("gauss", "clustered"):
        for q in (0.5, 0.75):
            sel = lambda r, kind=kind, q=q: r["kind"] == kind and r["q"] == q  # noqa: E731
            out["summary"][f"{kind} q={q}"] = {k: agg(k, sel) for k in ("self", "same_proj", "own_proj", "random")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
