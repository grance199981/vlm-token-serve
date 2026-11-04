"""Regression lock: the selection of every method is bitwise identical to the version that
produced the published results (internal plugin 0.1.16-0.1.19; hashes computed with the
original select.py). Any change to a selection rule, its tie-breaking or its arithmetic fails
here; the last test checks that the lock itself would notice such a change."""
import hashlib
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vts_plugin import select as S  # noqa: E402

HASHES = {"random": "a6de06bf99524a2b", "pool2x2": "d0b70d5f58ae0e12",
          "divprune": "a40683c0b77728e7", "rate": "9be6cd62e7716d10",
          "rate_pr": "9b9469bbaf64fd56", "rate_orig": "a49d5e56e30a8c47"}


def feats(n, d=64, seed=0, rank=8):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(rank, d, generator=g)
    mix = torch.randn(n, rank, generator=g)
    return mix @ base + 0.3 * torch.randn(n, d, generator=g)


def selection_hash(method, select_fn=S.select):
    h = hashlib.sha1()
    for n, gw in ((256, 16), (600, 24)):
        x = feats(n, seed=n)
        r = torch.arange(n)
        pos = torch.stack([torch.zeros_like(r), r // gw, r % gw], 1)
        _, p = select_fn(method, x, pos, k=n // 2, grid_h=n // gw, grid_w=gw, seed=0, score_dim=48)
        h.update(p.numpy().tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("method", sorted(HASHES))
def test_selection_unchanged(method):
    assert selection_hash(method) == HASHES[method]


def test_lock_detects_a_changed_rule():
    """Negative control: RATE with the opposite sort direction must not match the lock."""
    def flipped(method, x, pos, *, k, grid_h, grid_w, seed, score_dim):
        return S.select_rate(x, pos, k, seed, descending=False, score_dim=score_dim)

    assert selection_hash("rate", flipped) != HASHES["rate"]
