"""T4: unit tests for the selection methods (CPU, no vLLM needed)."""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from vts_plugin.config import VTSConfig, keep_count, parse_config  # noqa: E402
from vts_plugin.select import (  # noqa: E402
    divprune_indices,
    rate_indices,
    select,
    select_pool2x2,
)


def grid_pos(gh, gw):
    r = torch.arange(gh).repeat_interleave(gw)
    c = torch.arange(gw).repeat(gh)
    return torch.stack([torch.zeros_like(r), r, c, torch.full_like(r, gw), torch.full_like(r, gh)], 1)


def official_divprune(x, n_keep):
    """Verbatim logic of vbdi/divprune LLaVA/llava/model/llava_arch.py::DivPrune
    (threshold_terms replaced by n_keep)."""
    norm = x / x.norm(dim=1, keepdim=True)
    cosine_matrix = 1.0 - norm @ norm.t()
    s = torch.empty(n_keep, dtype=torch.long)
    for i in range(n_keep):
        if i == 0:
            m2 = cosine_matrix
            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]
        else:
            m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s, 0, torch.arange(0, i)))
            scores = torch.min(m2, dim=0).values
        s[i] = torch.argmax(scores)
    return s


@pytest.mark.parametrize("n,k,seed", [(64, 16, 0), (196, 49, 1), (784, 196, 2), (300, 37, 3)])
def test_divprune_matches_official(n, k, seed):
    x = torch.randn(n, 32, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).float()
    assert torch.equal(divprune_indices(x, k), official_divprune(x, k))


def divprune_indices_gemv(feats, k):
    """select.divprune_indices as of plugin 0.1.8 (1-D GEMV in the greedy loop)."""
    x = F.normalize(feats.float(), p=2, dim=-1)
    nn_dist = torch.empty(x.shape[0])
    for s in range(0, x.shape[0], 1024):
        nn_dist[s:s + 1024] = torch.topk(1.0 - x[s:s + 1024] @ x.T, 2, dim=1, largest=False).values[:, 1]
    sel = torch.empty(k, dtype=torch.long)
    sel[0] = torch.argmax(nn_dist)
    min_dist = 1.0 - x @ x[sel[0]]
    for i in range(1, k):
        sel[i] = torch.argmax(min_dist)
        min_dist = torch.minimum(min_dist, 1.0 - x @ x[sel[i]])
    return sel


@pytest.mark.parametrize("n,d,k,seed", [(784, 64, 196, 0), (1200, 2560, 600, 1), (2500, 128, 300, 2)])
def test_divprune_2d_form_matches_0_1_8(n, d, k, seed):
    # 0.1.9 replaced x @ v by (x @ v[:, None])[:, 0]; the picks must be unchanged
    x = torch.randn(n, d, generator=torch.Generator().manual_seed(seed))
    old, new = divprune_indices_gemv(x, k), divprune_indices(x, k)
    assert torch.equal(new, old)
    assert 0 < k < n and torch.unique(new).numel() == k


def test_divprune_negative_control():
    x = torch.randn(100, 16, generator=torch.Generator().manual_seed(0))
    # a different selection rule (first k) must not coincide with DivPrune
    assert not torch.equal(torch.sort(divprune_indices(x, 20))[0], torch.arange(20))


def pr_rate_mask(embeds, q, rand_proj, pivot_block_size=32, skip_ratio=0.01, descending=False):
    """Verbatim vLLM PR #60963 compute_retention_mask_rate / _get_diversity_scores
    (head 06158cbe2748) with the random projections injected; `descending=False`
    is the PR as written, `descending=True` our sort-direction fix."""
    num_tokens = embeds.shape[0]
    retain_num_tokens = max(1, int(num_tokens * (1 - q)))
    if num_tokens < pivot_block_size:
        scores = torch.ones(num_tokens, dtype=embeds.dtype)
    else:
        seq_norm = F.normalize(embeds, p=2, dim=-1)
        init_scores = torch.einsum("ik,jk->i", seq_norm, rand_proj)
        sorted_values, sorted_idx = init_scores.sort(dim=-1, descending=descending, stable=True)
        skip_num = max(1, int(skip_ratio * num_tokens))
        denom = sorted_values[skip_num] - sorted_values[-skip_num]
        if denom <= 0:
            scores = torch.zeros(num_tokens, dtype=embeds.dtype)
        else:
            q_scale = (num_tokens / pivot_block_size) / denom
            sorted_q = (sorted_values * q_scale).round()
            pivot_mask = sorted_q[:-1] > sorted_q[1:]
            pivot_idx = torch.cat([sorted_idx[:1], sorted_idx[1:][pivot_mask]])
            pivot_set = seq_norm[pivot_idx]
            scores = torch.einsum("ik,jk->i", seq_norm, pivot_set)
            scores /= pivot_set.shape[0]
            scores[pivot_idx] = 0.0
    order = torch.argsort(scores, descending=False, stable=True)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    mask[order[:retain_num_tokens]] = True
    return mask


def _rate_case(n, keep, seed=7):
    x = torch.randn(n, 64, generator=torch.Generator().manual_seed(n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = F.normalize(torch.randn(8, 64, generator=g), p=2, dim=-1)
    return x, proj, max(1, int(n * keep)), seed


@pytest.mark.parametrize("n,keep", [(784, 0.25), (784, 0.5), (400, 0.125)])
def test_rate_pr_as_written_keeps_a_prefix(n, keep):
    x, proj, k, seed = _rate_case(n, keep)
    ref = pr_rate_mask(x, 1 - keep, proj, descending=False)
    assert ref[:k].all() and not ref[k:].any(), "PR as written degenerates to prefix retention"
    ours = torch.zeros(n, dtype=torch.bool)
    ours[rate_indices(x, k, seed, descending=False)] = True
    assert torch.equal(ours, ref)


@pytest.mark.parametrize("n,keep", [(784, 0.25), (784, 0.5), (400, 0.125)])
def test_rate_fixed_matches_reference_and_differs_from_prefix(n, keep):
    x, proj, k, seed = _rate_case(n, keep)
    ref = pr_rate_mask(x, 1 - keep, proj, descending=True)
    ours = torch.zeros(n, dtype=torch.bool)
    ours[rate_indices(x, k, seed, descending=True)] = True
    assert torch.equal(ours, ref)
    assert not (ref[:k].all() and not ref[k:].any()), "fixed RATE must not be a prefix"


@pytest.mark.parametrize("method", ["random", "divprune", "rate", "rate_pr", "rate_orig", "pool2x2", "keep_all"])
@pytest.mark.parametrize("gh,gw,keep", [(28, 28, 0.25), (22, 40, 0.5), (7, 14, 0.125), (5, 3, 0.5), (1, 1, 0.25)])
def test_counts_and_order(method, gh, gw, keep):
    n = gh * gw
    cfg = VTSConfig(method=method, keep=keep, seed=0)
    k = keep_count(cfg, n, gh, gw)
    feats = torch.randn(n, 16, generator=torch.Generator().manual_seed(1))
    pos = grid_pos(gh, gw)
    f2, p2 = select(method, feats, pos, k=k, grid_h=gh, grid_w=gw, seed=0)
    assert f2.shape == (k, 16) and p2.shape == (k, 5)
    raster = p2[:, 1] * gw + p2[:, 2]
    assert torch.all(raster[1:] > raster[:-1]), "selected tokens must stay in raster order"
    if method == "pool2x2":
        assert k == math.ceil(gh / 2) * math.ceil(gw / 2)


def test_random_seed_determinism_and_negative_control():
    feats, pos = torch.randn(784, 8), grid_pos(28, 28)
    a = select("random", feats, pos, k=196, grid_h=28, grid_w=28, seed=0)[1]
    b = select("random", feats, pos, k=196, grid_h=28, grid_w=28, seed=0)[1]
    c = select("random", feats, pos, k=196, grid_h=28, grid_w=28, seed=1)[1]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_pool2x2_values_and_positions():
    gh, gw = 3, 3
    feats = torch.arange(9, dtype=torch.float32).unsqueeze(1)  # value = raster index
    f, p = select_pool2x2(feats, grid_pos(gh, gw), gh, gw)
    # blocks: {0,1,3,4} {2,5} {6,7} {8}
    assert f.squeeze(1).tolist() == [2.0, 3.5, 6.5, 8.0]
    assert (p[:, 1] * gw + p[:, 2]).tolist() == [0, 2, 6, 8]  # top-left tokens


def test_keep_all_is_identity():
    feats, pos = torch.randn(50, 4), grid_pos(5, 10)
    f, p = select("keep_all", feats, pos, k=50, grid_h=5, grid_w=10, seed=0)
    assert torch.equal(f, feats) and torch.equal(p, pos)


def test_parse_config_rejects_bad_values():
    with pytest.raises(ValueError):
        parse_config({"VTS_METHOD": "magic"})
    with pytest.raises(ValueError):
        parse_config({"VTS_METHOD": "random", "VTS_KEEP": "0"})
    assert not parse_config({}).enabled


def test_downscale_pixel_ratio_and_types():
    import numpy as np
    from PIL import Image

    from vts_plugin.config import VTSConfig, keep_count
    from vts_plugin.resize import downscale

    img = Image.new("RGB", (1280, 720))
    for keep in (0.5, 0.25, 0.125):
        w, h = downscale(img, keep).size
        assert abs(w * h / (1280 * 720) - keep) < 0.01
    assert downscale(img, 1.0) is img
    arr = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert downscale(arr, 0.25).shape == (360, 640, 3)
    cfg = VTSConfig(method="downsample", keep=0.25)
    assert cfg.enabled and not cfg.prunes  # pruning machinery stays off for B1
    assert keep_count(cfg, 784, 28, 28) == 784
    assert VTSConfig(method="random", keep=0.5).prunes


@pytest.mark.parametrize("method", ["divprune", "rate"])
def test_score_dim_uses_main_features_only_and_keeps_whole_rows(method):
    g = torch.Generator().manual_seed(3)
    main = torch.randn(196, 16, generator=g)
    extra = torch.randn(196, 48, generator=g)  # DeepStack-like columns
    pos = grid_pos(14, 14)
    full = torch.cat([main, extra], 1)
    f1, p1 = select(method, full, pos, k=49, grid_h=14, grid_w=14, seed=0, score_dim=16)
    f0, p0 = select(method, main, pos, k=49, grid_h=14, grid_w=14, seed=0)
    assert torch.equal(p1, p0), "selection must depend on the main columns only"
    assert f1.shape == (49, 64) and torch.equal(f1[:, :16], f0)
    # negative control: scoring on all columns picks a different set
    f2, p2 = select(method, full, pos, k=49, grid_h=14, grid_w=14, seed=0)
    assert not torch.equal(p2, p0)
