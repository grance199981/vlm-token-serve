"""DivPrune greedy max-min selection with one Triton kernel (VTS_DIVPRUNE_IMPL=triton).

The PyTorch version (select.divprune_indices) runs k dependent steps, each a GEMV over all N
tokens plus an argmax: about 4 small kernels per step, so ~1600 launches for one 784-token
image at keep 0.5 (~25 ms, launch bound). Here:
  1. S = x x^T once (cuBLAS, fp32; x = L2-normalised features), N x N;
  2. first pick on S exactly as the PyTorch version (argmax of the nearest-neighbour distance);
  3. ONE Triton program runs all k-1 remaining steps: the running min distance stays in
     registers, each step reads one row of S (L2 resident) and takes the argmax.
Same arithmetic per element as the PyTorch loop (min(min_dist, 1 - s), first maximal index on
ties); S comes from a GEMM instead of k GEMVs, so values can differ in the last bits and
near-ties can resolve differently (tests check a greedy certificate, not bitwise equality).
N > MAX_TOKENS falls back to the PyTorch version (S would need N^2 * 4 bytes).
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only test environments
    triton = None

MAX_TOKENS = 4096  # S = 64 MiB fp32; the register block holds next_pow2(N) <= 4096 floats

if triton is not None:
    @triton.jit
    def _greedy_maxmin_kernel(s_ptr, sel_ptr, first_ptr, n, k, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        valid = offs < n
        # running min distance to the selected set; -inf on padding so argmax never picks it
        md = tl.where(valid, float("inf"), float("-inf"))
        cur = tl.load(first_ptr).to(tl.int64)  # read on device: no host sync
        tl.store(sel_ptr, cur)
        for i in range(1, k):
            s = tl.load(s_ptr + cur * n + offs, mask=valid, other=0.0)
            md = tl.where(valid, tl.minimum(md, 1.0 - s), float("-inf"))
            cur = tl.argmax(md, axis=0, tie_break_left=True).to(tl.int64)
            tl.store(sel_ptr + i, cur)


def similarity(feats):
    x = F.normalize(feats.float(), p=2, dim=-1)
    return x @ x.T


def first_pick(sim):
    """argmax over tokens of the distance to the nearest other token (as divprune_indices);
    a 0-d device tensor."""
    nn_dist = torch.topk(1.0 - sim, 2, dim=1, largest=False).values[:, 1]
    return torch.argmax(nn_dist)


def greedy_from_sim_torch(sim, k, first):
    """Reference for the kernel on the same S (pure PyTorch, same per-element arithmetic)."""
    n = sim.shape[0]
    sel = torch.empty(k, dtype=torch.long, device=sim.device)
    sel[0] = first
    md = torch.full((n,), float("inf"), device=sim.device)
    for i in range(1, k):
        md = torch.minimum(md, 1.0 - sim[sel[i - 1]])
        sel[i] = torch.argmax(md)
    return sel


def greedy_from_sim_triton(sim, k, first):
    n = sim.shape[0]
    assert sim.is_cuda and sim.dtype == torch.float32 and sim.is_contiguous()
    sel = torch.empty(k, dtype=torch.int64, device=sim.device)
    block = max(16, triton.next_power_of_2(n))
    _greedy_maxmin_kernel[(1,)](sim, sel, first.reshape(1), n, k, BLOCK=block,
                                num_warps=4 if block <= 1024 else 8)
    return sel


def available(feats):
    return triton is not None and feats.is_cuda and feats.shape[0] <= MAX_TOKENS


def divprune_indices_triton(feats, k):
    n = feats.shape[0]
    if k >= n:
        return torch.arange(n, device=feats.device)
    sim = similarity(feats).contiguous()
    return greedy_from_sim_triton(sim, k, first_pick(sim))
