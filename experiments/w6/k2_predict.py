"""Pre-registered K2 prediction (docs/plans/K2_PLAN.md §5), from the B0 calibration and the block ledger.

A session's turn t+1 hits its cached prefix iff the cache still holds all S sessions' turn-t chains:
S * B_t <= C - O, with B_t the cached blocks of one session after turn t and O the blocks held by the
8 in-flight requests beyond what is cached (fitted per model from the B0 calibration, then fixed).
  dense (Qwen3-VL, 16-token blocks):  B_t = floor((P_t + out) / 16); a hit covers 16 * B_t tokens.
  hybrid (Qwen3.5, 528-token blocks): B_t = 4 * floor((P_t + out) / 528) (1 attention + 3 GDN state
    blocks per full block); a hit covers floor((P_t - think) / 528) * 528 tokens, because the next
    turn re-renders this turn's reply without the generation prompt's think tags.
P_t = text tokens of turn t + vision tokens (784 for 896x896; 400 downsample-50%; 392 pruning-50%).
The predicted turn 2-4 hit-rate curve is evaluated on the formal grid and S* is read off it with the
same interpolation as k2_summary.py.

Usage: python k2_predict.py [--json pred.json]
"""
import argparse
import json
import math

OUT = 64
GRID = [8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 72, 80, 96]
# B0 calibration run (K2 §5): mean prompt tokens per turn and pool size
MODELS = {
    "Qwen3-VL-4B-Instruct": {"text": [28, 117, 206, 295], "C": 1820, "O": 65, "kind": "dense"},
    "Qwen3.5-4B": {"text": [32, 121, 210, 299], "C": 248, "O": 56, "kind": "hybrid", "think": 5},
}
VISION = {"none": 784, "downsample": 400, "random": 392}


def ledger(model, method):
    m = MODELS[model]
    P = [t + VISION[method] for t in m["text"]]
    steps = []  # (turn t+1, S_max, hit tokens, prompt tokens of turn t+1)
    for t in range(3):
        if m["kind"] == "dense":
            b = (P[t] + OUT) // 16
            hit = 16 * b
        else:
            b = 4 * ((P[t] + OUT) // 528)
            hit = (P[t] - m["think"]) // 528 * 528
        steps.append((t + 2, (m["C"] - m["O"]) / b if b else math.inf, hit, P[t + 1]))
    return P, steps


def curve(steps):
    tot = sum(s[3] for s in steps)
    return [(S, sum(h for _, smax, h, _ in steps if S <= smax) / tot) for S in GRID]


def s_star(c):
    (s0, h0), half = c[0], c[0][1] / 2
    if h0 <= 0:
        return float("nan")
    for (sa, ha), (sb, hb) in zip(c, c[1:]):
        if hb <= half:
            return sa + (ha - half) / (ha - hb) * (sb - sa)
    return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    pred = {}
    for model in MODELS:
        b0 = None
        for method in VISION:
            P, steps = ledger(model, method)
            c = curve(steps)
            s = s_star(c)
            b0 = s if method == "none" else b0
            pred.setdefault(model, {})[method] = s
            print(f"{model} {method}: P_t={P}; S_max per turn " +
                  ", ".join(f"t{t}={smax:.1f} (hit {h}/{p})" for t, smax, h, p in steps) +
                  f"; hit at S=8 {100 * c[0][1]:.1f}%; S* {s:.1f}; gain {s / b0:.3f}")
            print("   curve: " + " ".join(f"{S}:{100 * h:.1f}" for S, h in c))
    if a.json:
        json.dump(pred, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
