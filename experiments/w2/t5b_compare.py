"""T5b: HF reference vs vLLM plugin on 100 single-image requests (pass 0).

Confident disagreement at the first divergent token: one side's chosen token has
p >= 0.8 while the token the other side chose has p <= 0.05 on that side (a token
outside the top-5 counts as p <= 0.05; with top-1 >= 0.8 the 5th is below 0.05
in practice). Near-tie flips do not count.

Criterion (fixed before the run): confident(random pair) <= confident(base pair) + 2.
Diagnostic: the same count for HF random50 with placeholder positions; if vLLM
agrees with that variant better than with the correct-position variant, vLLM is
not applying the recomputed positions.
"""
import json
import math
import os
import sys

HI, LO = math.log(0.8), math.log(0.05)


def load(p):
    return {x["req"]: x for x in json.load(open(p)) if x["rep"] == 0}


def confident(a, b):
    for pos, ((ta, la), (tb, lb)) in enumerate(zip(a["logprobs"], b["logprobs"])):
        if ta == tb:
            continue
        a_top, b_top = dict(a["top_logprobs"][pos]), dict(b["top_logprobs"][pos])
        a_conf = la >= HI and a_top.get(tb, -99.0) <= LO
        b_conf = lb >= HI and b_top.get(ta, -99.0) <= LO
        return a_conf or b_conf
    return False


def main(hf_dir, vllm_dir, model):
    pairs = {"base": ("hf_keep_all", "vanilla_nopc"), "random": ("hf_random50", "random50_nopc"),
             "random_vs_hf_placeholder": ("hf_random50_placeholder", "random50_nopc")}
    out = {}
    print("| model | pair | text differs | confident disagreements |\n|---|---|---|---|")
    for name, (h, v) in pairs.items():
        hp, vp = os.path.join(hf_dir, f"{model}_{h}.json"), os.path.join(vllm_dir, f"{model}_{v}.json")
        if not (os.path.exists(hp) and os.path.exists(vp)):
            print(f"| {model} | {name} | missing | |")
            continue
        hf, vl = load(hp), load(vp)
        diff = [q for q in hf if hf[q]["output"].strip() != vl[q]["output"].strip()]
        conf = [q for q in hf if confident(hf[q], vl[q])]
        out[name] = conf
        print(f"| {model} | {name} | {len(diff)}/{len(hf)} | {len(conf)} {conf[:12]} |")
    ok = "random" in out and "base" in out and len(out["random"]) <= len(out["base"]) + 2
    print("T5b", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
