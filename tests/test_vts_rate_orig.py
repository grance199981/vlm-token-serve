"""`rate_orig` must select exactly what PR #60963 head 45333ba selects (CPU, no vLLM needed).

Reference: tests/fixtures/rate_pr_45333ba.py, a verbatim copy of the PR's rate.py. For each
input we compare the full retention mask, not just the count. Negative controls: our
descending variant (`rate`) and the pre-fix PR (`rate_pr`, a prefix) must NOT match the
reference on the same inputs, so the equality checks are not passing vacuously.
"""
import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

from vts_plugin.config import VTSConfig, keep_count  # noqa: E402
from vts_plugin.select import rate_indices, rate_orig_indices, select  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rate_pr_45333ba", pathlib.Path(__file__).parent / "fixtures" / "rate_pr_45333ba.py")
PR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PR)


def feats(kind, n, d, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    if kind == "gauss":
        x = torch.randn(n, d, generator=g)
    elif kind == "clustered":  # unequal clusters plus noise, closer to image features
        c = torch.randn(12, d, generator=g)
        w = torch.rand(12, generator=g) ** 2 + 0.05
        x = c[torch.multinomial(w / w.sum(), n, replacement=True, generator=g)]
        x = x + 0.6 * torch.randn(n, d, generator=g)
    elif kind == "constant":  # every token identical: denom == 0 branch
        x = torch.randn(1, d, generator=g).expand(n, d).clone()
    else:
        raise ValueError(kind)
    return x.to(dtype)


def mask_of(idx, n):
    m = torch.zeros(n, dtype=torch.bool)
    m[idx] = True
    return m


CASES = [(kind, n, d, seed, dtype)
         for kind in ("gauss", "clustered")
         for n, d in ((784, 64), (196, 2560), (3136, 128), (1200, 10240), (40, 32))
         for seed in (0, 1, 2)
         for dtype in (torch.float32, torch.bfloat16)]


@pytest.mark.parametrize("keep", [0.5, 0.25])
@pytest.mark.parametrize("kind,n,d,seed,dtype", CASES)
def test_rate_orig_matches_pr_head(kind, n, d, seed, dtype, keep):
    x = feats(kind, n, d, seed, dtype)
    k = keep_count(VTSConfig(method="rate_orig", keep=keep), n, 1, n)
    ref = PR.compute_retention_mask_rate(x, q=1.0 - keep)
    assert int(ref.sum()) == k, "keep count must equal the PR's compute_retained_tokens_count"
    assert torch.equal(mask_of(rate_orig_indices(x, k), n), ref)


@pytest.mark.parametrize("n,keep", [(16, 0.5), (31, 0.25), (784, 0.5)])
def test_rate_orig_degenerate_inputs_match(n, keep):
    # n < pivot_block_size (prefix fallback) and identical tokens (denom == 0)
    for kind in ("gauss", "constant"):
        x = feats(kind, n, 64, 0)
        k = keep_count(VTSConfig(method="rate_orig", keep=keep), n, 1, n)
        assert torch.equal(mask_of(rate_orig_indices(x, k), n),
                           PR.compute_retention_mask_rate(x, q=1.0 - keep))


def test_negative_controls_differ_from_pr_head():
    n_desc_diff = n_prefix_diff = 0
    for kind, n, d, seed, dtype in CASES:
        if n < 32:
            continue
        x = feats(kind, n, d, seed, dtype)
        k = n // 2
        ref = PR.compute_retention_mask_rate(x, q=0.5)
        n_desc_diff += not torch.equal(mask_of(rate_indices(x, k, 0, descending=True), n), ref)
        n_prefix_diff += not torch.equal(mask_of(torch.arange(k), n), ref)
    n_cases = sum(1 for c in CASES if c[1] >= 32)
    assert n_desc_diff == n_cases, "our descending variant must not coincide with the PR head"
    assert n_prefix_diff == n_cases, "the PR head must not keep a prefix"


def test_select_rate_orig_ignores_score_dim():
    # like the PR, rate_orig scores the whole row (Qwen3-VL: main + DeepStack columns)
    gh, gw = 20, 30
    n = gh * gw
    x = feats("clustered", n, 4 * 64, 3)
    r = torch.arange(gh).repeat_interleave(gw)
    c = torch.arange(gw).repeat(gh)
    pos = torch.stack([torch.zeros_like(r), r, c, torch.full_like(r, gw), torch.full_like(r, gh)], 1)
    k = n // 4
    _, p_full = select("rate_orig", x, pos, k=k, grid_h=gh, grid_w=gw, seed=0, score_dim=64)
    ref = PR.compute_retention_mask_rate(x, q=0.75)
    assert torch.equal(mask_of(p_full[:, 1] * gw + p_full[:, 2], n), ref)
    sub = PR.compute_retention_mask_rate(x[:, :64], q=0.75)
    assert not torch.equal(sub, ref), "control: scoring on the first 64 columns selects differently"
