"""Pre-registered K1 prediction (docs/plans/W4_PLAN.md §4): admission-concurrency gain of
token reduction = time-averaged per-request KV footprint of B0 / that of the method.

Dense (Qwen3-VL-4B): footprint = ceil(tokens / 16) blocks (144 KiB per token).
Hybrid (Qwen3.5-4B): footprint = ceil(tokens / 528) attention blocks + 3 * m state
blocks (3 GDN groups, m physical blocks per group in vLLM's 'align' mamba cache
mode; m is fixed from the B0 calibration run, bounds 1..2 from the source code).
Average over the decode phase (tokens = prompt + 1 .. prompt + output).

Usage: python k1_predict.py [--text 50] [--output 256] [--images 2] [--keep 0.5] [--side 896] [--m 1 2]
"""
import argparse
import math

PATCH, MERGE = 16, 2


def vision_tokens(method, keep, images, SIDE=896):
    per = (SIDE // (PATCH * MERGE)) ** 2  # 784 for 896x896
    if method == "B0":
        return per * images
    if method == "downsample":  # resize by sqrt(keep), Qwen smart_resize rounds to 32
        side = round(SIDE * math.sqrt(keep) / (PATCH * MERGE)) * PATCH * MERGE
        return (side // (PATCH * MERGE)) ** 2 * images
    return max(1, int(per * keep)) * images  # random / rate / divprune keep_count


def avg_blocks(prompt, output, block, fixed):
    return sum(math.ceil((prompt + t) / block) + fixed for t in range(1, output + 1)) / output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=int, default=50, help="non-vision prompt tokens incl. template")
    ap.add_argument("--output", type=int, default=256)
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--side", type=int, default=896, help="square image side (multiple of 32)")
    ap.add_argument("--m", type=float, nargs="+", default=[1, 2])
    a = ap.parse_args()
    rows = []
    for method in ("B0", "downsample", "pruning"):
        v = vision_tokens(method, a.keep, a.images, a.side)
        rows.append((method, v, a.text + v))
    print(f"text={a.text} output={a.output} images={a.images} keep={a.keep} side={a.side}")
    print("| method | vision tokens | prompt tokens | dense gain | " +
          " | ".join(f"hybrid gain (m={m:g})" for m in a.m) + " |")
    print("|---|---|---|---|" + "---|" * len(a.m))
    b0 = rows[0][2]
    for method, v, p in rows:
        dense = avg_blocks(b0, a.output, 16, 0) / avg_blocks(p, a.output, 16, 0)
        hyb = [avg_blocks(b0, a.output, 528, 3 * m) / avg_blocks(p, a.output, 528, 3 * m) for m in a.m]
        print(f"| {method} | {v} | {p} | {dense:.3f} | " + " | ".join(f"{h:.3f}" for h in hyb) + " |")


if __name__ == "__main__":
    main()
