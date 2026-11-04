"""W5(a) operator micro-benchmark: DivPrune selection, PyTorch loop vs one Triton kernel.

For each token count N (one image's merged grid; 784 = one 896x896 image on both models) and
keep ratio: time select.divprune_indices (PyTorch) and divprune_triton.divprune_indices_triton
(GEMM + top-2 + one kernel; also each part alone), CUDA events, median of --reps after
--warmup. Inputs: clustered synthetic features, width 2560 (the main merger output that
DivPrune scores on), bf16 as in the plugin. Also reports agreement with the PyTorch version
(Jaccard of the two kept sets, identical sequence or not).

Usage: python divprune_bench.py --out <json> [--reps 30] [--warmup 5]
"""
import argparse
import json
import statistics

import torch

from vts_plugin import divprune_triton as dt
from vts_plugin.select import divprune_indices

NS = (196, 784, 1600, 3136, 4096)
KEEPS = (0.5, 0.25)
D = 2560


def feats(n, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randn(12, D, generator=g)
    w = torch.rand(12, generator=g) ** 2 + 0.05
    x = c[torch.multinomial(w / w.sum(), n, replacement=True, generator=g)]
    return (x + 0.6 * torch.randn(n, D, generator=g)).to(torch.bfloat16).cuda()


def timed(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()
    rows = []
    print("| N | k | PyTorch ms | Triton total ms | speedup | GEMM | top-2 + argmax | kernel | Jaccard | same sequence |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in NS:
        x = feats(n, n)
        for keep in KEEPS:
            k = int(n * keep)
            ref = timed(lambda: divprune_indices(x, k), a.reps, a.warmup)
            tri = timed(lambda: dt.divprune_indices_triton(x, k), a.reps, a.warmup)
            sim = dt.similarity(x).contiguous()
            first = dt.first_pick(sim)
            t_gemm = timed(lambda: dt.similarity(x), a.reps, a.warmup)
            t_first = timed(lambda: dt.first_pick(sim), a.reps, a.warmup)
            t_kern = timed(lambda: dt.greedy_from_sim_triton(sim, k, first), a.reps, a.warmup)
            s_ref, s_tri = divprune_indices(x, k), dt.divprune_indices_triton(x, k)
            A, B = set(s_ref.tolist()), set(s_tri.tolist())
            row = {"n": n, "k": k, "torch_ms": ref, "triton_ms": tri, "speedup": ref[0] / tri[0],
                   "gemm_ms": t_gemm, "first_ms": t_first, "kernel_ms": t_kern,
                   "jaccard": len(A & B) / len(A | B), "same_sequence": bool(torch.equal(s_ref, s_tri))}
            rows.append(row)
            print(f"| {n} | {k} | {ref[0]:.2f} | {tri[0]:.3f} | {row['speedup']:.1f}x | {t_gemm[0]:.3f} | "
                  f"{t_first[0]:.3f} | {t_kern[0]:.3f} | {row['jaccard']:.3f} | {row['same_sequence']} |")
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
           "triton": dt.triton.__version__, "width": D, "dtype": "bfloat16", "reps": a.reps,
           "warmup": a.warmup, "rows": rows}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
