"""T5 comparison: HF reference (hf_prune_reference.py) vs vLLM plugin outputs.

Pairs (pass 0 only, 13 requests):
  base  : HF keep_all  vs vLLM vanilla_nopc   (no pruning: kernel/numeric gap)
  random: HF random50  vs vLLM random50_nopc
  pool  : HF pool2x2   vs vLLM pool2x2
For each differing request, margin = HF logprob(top-1) - HF logprob(token vLLM
chose) at the first divergent position (inf if not in HF top-5).

Pass criteria (fixed before the run):
  #differ(pair) <= #differ(base) + 2
  max margin(pair) <= max(1.0, max margin(base))
  and prompt_tokens equal between HF and vLLM for every request
"""
import json
import os
import sys


def load(p):
    return {x["req"]: x for x in json.load(open(p)) if x["rep"] == 0}


def margin(hf, vl):
    for pos, ((th, lh), (tv, _)) in enumerate(zip(hf["logprobs"], vl["logprobs"])):
        if th != tv:
            top = dict(hf["top_logprobs"][pos])
            return lh - top[tv] if tv in top else float("inf")
    return 0.0


def stats(hf, vl):
    diff = [q for q in hf if hf[q]["output"].strip() != vl[q]["output"].strip()]
    margins = [margin(hf[q], vl[q]) for q in diff]
    pt_ok = all(hf[q]["prompt_tokens"] == vl[q]["prompt_tokens"] for q in hf)
    return diff, margins, pt_ok


def main(hf_dir, vllm_dir, model):
    ok = True
    pairs = {"base": ("keep_all", "vanilla_nopc"), "random": ("random50", "random50_nopc"), "pool": ("pool2x2", "pool2x2")}
    res = {}
    print("| model | pair | text differs | margins at divergence | prompt_tokens equal | result |\n|---|---|---|---|---|---|")
    for name, (h, v) in pairs.items():
        hp, vp = os.path.join(hf_dir, f"{model}_hf_{h}.json"), os.path.join(vllm_dir, f"{model}_{v}.json")
        if not (os.path.exists(hp) and os.path.exists(vp)):
            print(f"| {model} | {name} | missing | | | FAIL |")
            ok = False
            continue
        res[name] = stats(load(hp), load(vp))
    base_diff, base_m, _ = res.get("base", ([], [], True))
    base_max = max(base_m) if base_m else 0.0
    for name, (diff, margins, pt_ok) in res.items():
        mx = max(margins) if margins else 0.0
        if name == "base":
            passed = pt_ok
        else:
            passed = pt_ok and len(diff) <= len(base_diff) + 2 and mx <= max(1.0, base_max)
        ok &= passed
        print(f"| {model} | {name} | {len(diff)}/13 {diff} | {[round(m, 3) for m in margins]} | {pt_ok} | {'PASS' if passed else 'FAIL'} |")
    print("T5", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
