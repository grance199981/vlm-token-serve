"""Visual-token selection methods (pure torch, no vLLM imports).

Every method takes the post-merger features of ONE image, `feats` of shape
(N, D), and its position channels `pos` of shape (N, C) whose columns 1 and 2
are the row / column of each token on the merged grid. It returns
(new_feats, new_pos) with exactly `k` rows (pool2x2: one row per 2x2 block), in
the original raster order, so the M-RoPE recompute sees a sorted token list.

Content-dependent methods (DivPrune, RATE) score tokens on the first `score_dim`
columns only: the main merger output that becomes the LLM input embedding. On
Qwen3-VL each row also carries 3 DeepStack levels (4 x 2560 columns); scoring on
all of them cost float32 memory outside vLLM's budget and caused a CUDA OOM on
large DocVQA images. Selected rows are kept whole.

Entry point: `select(method, feats, pos, ...)`, dispatching through `_METHODS`.
"""
from collections.abc import Callable

import torch
import torch.nn.functional as F

from . import divprune_triton

# calls of the Triton DivPrune path / fallbacks to PyTorch (N too large or no CUDA), for logs
DIVPRUNE_TRITON_STATS = {"triton": 0, "fallback": 0}


def _gather(feats: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor):
    """Rows `idx` of feats / pos, re-sorted into raster order."""
    idx, _ = torch.sort(idx)
    return feats[idx], pos[idx]


def _scored(feats: torch.Tensor, score_dim: int | None) -> torch.Tensor:
    return feats[:, :score_dim] if score_dim else feats


# --- random / pooling -------------------------------------------------------------------------

def select_random(feats, pos, k: int, seed: int):
    """Fixed random pattern per (seed, N): deterministic across runs and batches."""
    n = feats.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + n)
    idx = torch.randperm(n, generator=g)[:k].to(feats.device)
    return _gather(feats, pos, idx)


def select_pool2x2(feats, pos, grid_h: int, grid_w: int):
    """Average 2x2 blocks of the merged grid (edge blocks may be 1x2/2x1/1x1).
    The pooled token keeps the position of the block's top-left token."""
    rows = pos[:, 1].long()
    cols = pos[:, 2].long()
    bw = (grid_w + 1) // 2
    block = (rows // 2) * bw + (cols // 2)
    nb = ((grid_h + 1) // 2) * bw
    summed = torch.zeros(nb, feats.shape[1], dtype=torch.float32, device=feats.device)
    summed.index_add_(0, block, feats.float())
    counts = torch.bincount(block, minlength=nb).clamp_min(1).unsqueeze(1)
    pooled = (summed / counts).to(feats.dtype)
    # top-left token of each block = first token of the block in raster order
    order = torch.arange(feats.shape[0], device=feats.device)
    first = torch.full((nb,), feats.shape[0], dtype=torch.long, device=feats.device)
    first.scatter_reduce_(0, block, order, reduce="amin")
    return pooled, pos[first]


# --- DivPrune ---------------------------------------------------------------------------------

def divprune_indices(feats: torch.Tensor, k: int) -> torch.Tensor:
    """DivPrune (greedy max-min diversity), following the official implementation
    (vbdi/divprune, LLaVA/llava/model/llava_arch.py `DivPrune`):
      distance = 1 - cosine similarity
      first pick = argmax over tokens of the distance to its nearest neighbour
      next picks = argmax of the min distance to the already selected set
    Computed incrementally (O(N*k) time, O(N) extra memory) instead of materialising N x N.
    Returns the k picked indices in pick order."""
    x = F.normalize(feats.float(), p=2, dim=-1)
    n = x.shape[0]
    if k >= n:
        return torch.arange(n, device=feats.device)
    # nearest-neighbour distance per token (second smallest, self distance ~ 0)
    nn_dist = torch.empty(n, device=x.device)
    for s in range(0, n, 1024):
        d = 1.0 - x[s:s + 1024] @ x.T
        nn_dist[s:s + 1024] = torch.topk(d, 2, dim=1, largest=False).values[:, 1]
    sel = torch.empty(k, dtype=torch.long, device=x.device)
    sel[0] = torch.argmax(nn_dist)
    # (N, d) @ (d, 1) rather than a GEMV with a 1-D vector: vLLM's batch-invariant aten::matmul
    # override (VLLM_BATCH_INVARIANT=1) only accepts 2-D operands. Same picks as the 1-D form.
    min_dist = 1.0 - (x @ x[sel[0]].unsqueeze(1)).squeeze(1)
    for i in range(1, k):
        sel[i] = torch.argmax(min_dist)
        min_dist = torch.minimum(min_dist, 1.0 - (x @ x[sel[i]].unsqueeze(1)).squeeze(1))
    return sel


def select_divprune(feats, pos, k: int, score_dim: int | None = None, impl: str = "torch"):
    """DivPrune on the first `score_dim` columns; `impl="triton"` uses the fused kernel
    when the image fits in it and falls back to the PyTorch loop otherwise."""
    scored = _scored(feats, score_dim)
    if impl == "triton":
        if divprune_triton.available(scored):
            DIVPRUNE_TRITON_STATS["triton"] += 1
            return _gather(feats, pos, divprune_triton.divprune_indices_triton(scored, k))
        DIVPRUNE_TRITON_STATS["fallback"] += 1
    return _gather(feats, pos, divprune_indices(scored, k))


# --- RATE: ported from vLLM PR #60963 (vllm/multimodal/image_prune/rate.py,
# head 06158cbe2748, Apache-2.0, Copyright contributors to the vLLM project).
# Changes: takes k directly (so the count always equals keep_count), uses a
# seeded generator for the random projections (the PR uses the global RNG, so
# its selection changes between server restarts), and is fed features only.
#
# Sort direction: the PR sorts the projection scores ascending, which makes
# `denom = v[skip] - v[-skip]` negative for every real input, so the PR always
# returns constant scores and keeps a PREFIX of the tokens (verified on the
# verbatim PR file, 2026-10-20). The bucket logic that follows
# (`q_scale = buckets / denom`, boundaries where `q[:-1] > q[1:]`) only works
# with a descending sort, so `rate` uses descending (our reading of the intent)
# and `rate_pr` keeps the PR behaviour for reproduction.

def _rate_scores(embeds, pivot_block_size: int, skip_num: int, num_proj: int, seed: int,
                 descending: bool = True) -> torch.Tensor:
    """Per-token score = mean cosine similarity to the bucket pivots (pivots score 0).
    Lower = more distinct = kept first."""
    num_tokens = embeds.shape[0]
    if num_tokens < pivot_block_size:
        return torch.ones(num_tokens, dtype=embeds.dtype, device=embeds.device)
    seq_norm = F.normalize(embeds, p=2, dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand_proj = F.normalize(
        torch.randn(num_proj, embeds.shape[-1], generator=g).to(embeds.dtype).to(embeds.device),
        p=2, dim=-1)
    init_scores = torch.einsum("ik,jk->i", seq_norm, rand_proj)
    sorted_values, sorted_idx = init_scores.sort(dim=-1, descending=descending, stable=True)
    denom = sorted_values[skip_num] - sorted_values[-skip_num]
    if denom <= 0:
        return torch.zeros(num_tokens, dtype=embeds.dtype, device=embeds.device)
    q_scale = (num_tokens / pivot_block_size) / denom
    sorted_q = (sorted_values * q_scale).round()
    pivot_mask = sorted_q[:-1] > sorted_q[1:]
    pivot_idx = torch.cat([sorted_idx[:1], sorted_idx[1:][pivot_mask]])
    pivot_set = seq_norm[pivot_idx]
    scores = torch.einsum("ik,jk->i", seq_norm, pivot_set)
    scores /= pivot_set.shape[0]
    scores[pivot_idx] = 0.0
    return scores


def rate_indices(feats, k: int, seed: int, pivot_block_size: int = 32, skip_ratio: float = 0.01,
                 num_proj: int = 8, descending: bool = True) -> torch.Tensor:
    n = feats.shape[0]
    if k >= n:
        return torch.arange(n, device=feats.device)
    scores = _rate_scores(feats, pivot_block_size, max(1, int(skip_ratio * n)), num_proj, seed,
                          descending)
    order = torch.argsort(scores, descending=False, stable=True)
    return order[:k]


def select_rate(feats, pos, k: int, seed: int, descending: bool = True,
                score_dim: int | None = None):
    return _gather(feats, pos, rate_indices(_scored(feats, score_dim), k, seed,
                                            descending=descending))


# `rate_orig`: RATE as in PR #60963 head 45333ba (the author's 2026-10-22 fix: ascending sort,
# guard only `denom == 0`, fixed-seed projections), copied line for line from
# vllm/multimodal/image_prune/rate.py at that commit. With an ascending sort `denom` is
# negative, so each bucket's pivot is its lowest-score token (ours, `rate`, takes the highest).
# Like the PR it scores the whole row before the position channels, i.e. on Qwen3-VL the
# DeepStack columns too (no `score_dim`), in the input dtype.
# The duplication with `_rate_scores` is deliberate: this copy must stay identical to the PR.

def _rate_orig_projections(num_proj, embed_dim, dtype, device):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0x9E3779B9 + embed_dim * 0x517CC1B7 + num_proj)
    return F.normalize(
        torch.randn(num_proj, embed_dim, dtype=torch.float32, generator=gen), p=2, dim=-1
    ).to(device=device, dtype=dtype)


def _rate_orig_scores(embeds, pivot_block_size, skip_num, num_proj):
    num_tokens = embeds.shape[0]
    if num_tokens < pivot_block_size:
        return torch.ones(num_tokens, dtype=embeds.dtype, device=embeds.device)
    seq_norm = F.normalize(embeds, p=2, dim=-1)
    rand_proj = _rate_orig_projections(num_proj, embeds.shape[-1], embeds.dtype, embeds.device)
    init_scores = torch.einsum("ik,jk->i", seq_norm, rand_proj)
    sorted_values, sorted_idx = init_scores.sort(dim=-1, stable=True)
    denom = sorted_values[skip_num] - sorted_values[-skip_num]
    if denom == 0:
        return torch.zeros(num_tokens, dtype=embeds.dtype, device=embeds.device)
    q_scale = (num_tokens / pivot_block_size) / denom
    sorted_q = (sorted_values * q_scale).round()
    pivot_mask = sorted_q[:-1] > sorted_q[1:]
    pivot_idx = torch.cat([sorted_idx[:1], sorted_idx[1:][pivot_mask]])
    pivot_set = seq_norm[pivot_idx]
    scores = torch.einsum("ik,jk->i", seq_norm, pivot_set)
    scores /= pivot_set.shape[0]
    scores[pivot_idx] = 0.0
    return scores


def rate_orig_indices(feats, k: int, pivot_block_size: int = 32, skip_ratio: float = 0.01,
                      num_proj: int = 8) -> torch.Tensor:
    n = feats.shape[0]
    if k >= n:
        return torch.arange(n, device=feats.device)
    scores = _rate_orig_scores(feats, pivot_block_size, max(1, int(skip_ratio * n)), num_proj)
    order = torch.argsort(scores, descending=False, stable=True)
    return order[:k]


def select_rate_orig(feats, pos, k: int):
    return _gather(feats, pos, rate_orig_indices(feats, k))


# --- dispatch ---------------------------------------------------------------------------------

_Selector = Callable[..., tuple[torch.Tensor, torch.Tensor]]

_METHODS: dict[str, _Selector] = {
    "keep_all": lambda f, p, a: (f, p),
    "random": lambda f, p, a: select_random(f, p, a["k"], a["seed"]),
    "pool2x2": lambda f, p, a: select_pool2x2(f, p, a["grid_h"], a["grid_w"]),
    "divprune": lambda f, p, a: select_divprune(f, p, a["k"], a["score_dim"], a["divprune_impl"]),
    "rate": lambda f, p, a: select_rate(f, p, a["k"], a["seed"], descending=True,
                                        score_dim=a["score_dim"]),
    "rate_pr": lambda f, p, a: select_rate(f, p, a["k"], a["seed"], descending=False,
                                           score_dim=a["score_dim"]),
    "rate_orig": lambda f, p, a: select_rate_orig(f, p, a["k"]),
}


def select(method: str, feats: torch.Tensor, pos: torch.Tensor, *, k: int, grid_h: int,
           grid_w: int, seed: int = 0, score_dim: int | None = None,
           divprune_impl: str = "torch") -> tuple[torch.Tensor, torch.Tensor]:
    """Select the tokens of one image with `method` (one of config.PRUNING_METHODS).

    feats: (N, D) post-merger features; pos: (N, C) position channels (see module doc).
    k: tokens to keep (config.keep_count); grid_h, grid_w: merged grid size (N = grid_h*grid_w).
    seed: random / RATE projections; score_dim: columns used for scoring (None = all).
    """
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(f"unknown selection method {method!r}") from None
    args = dict(k=k, grid_h=grid_h, grid_w=grid_w, seed=seed, score_dim=score_dim,
                divprune_impl=divprune_impl)
    return fn(feats, pos, args)
