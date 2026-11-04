"""Near-tie analysis of greedy divergences between two runs.

At the first generated position where runs A and B pick different tokens, report
the margin = logprob(top-1) - logprob(token B picked) in run A's top-5 list (and
vice versa). Small margins (< 0.5 nats) mean the two runs split a near-tie, i.e.
numeric noise; large margins point to a real difference in the computation.

Usage: python experiments/w2/divergence_margin.py <out_dir> <model> <runA> <runB>
"""
import json
import os
import sys


def load(d, m, n):
    return {(x["rep"], x["req"]): x for x in json.load(open(os.path.join(d, f"{m}_{n}.json")))}


def margin(x, pos, token):
    top = dict(x["top_logprobs"][pos])
    best = x["logprobs"][pos][1]
    return best - top[token] if token in top else float("inf")


def main(d, m, a, b):
    A, B = load(d, m, a), load(d, m, b)
    for k in sorted(A):
        x, y = A[k], B[k]
        if x["output"] == y["output"]:
            continue
        pos = next((i for i, (p, q) in enumerate(zip(x["logprobs"], y["logprobs"])) if p[0] != q[0]), None)
        if pos is None:
            print(f"{k}: outputs differ but token streams equal up to the shorter length")
            continue
        ma, mb = margin(x, pos, y["logprobs"][pos][0]), margin(y, pos, x["logprobs"][pos][0])
        print(f"{k}: first divergence at token {pos}: {x['logprobs'][pos][0]!r} vs {y['logprobs'][pos][0]!r}; "
              f"margin in {a} {ma:.3f}, in {b} {mb:.3f} -> {'near-tie' if max(ma, mb) < 0.5 else 'LARGE'}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
