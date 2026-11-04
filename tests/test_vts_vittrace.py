"""ViT call trace (vittrace.py): one JSON line per call with the right image / patch counts, the
wrapped forward's result unchanged, idempotent install; nothing patched when the env var is unset."""
import json
import os

import pytest

torch = pytest.importorskip("torch")

from vts_plugin import vittrace  # noqa: E402


class FakeViT:
    def forward(self, x, grid_thw, *, encoder_metadata=None):
        return x * 2


def test_trace_records_each_call(tmp_path):
    cls = type("FakeViT1", (FakeViT,), {})
    assert vittrace.install(cls, str(tmp_path))
    assert not vittrace.install(cls, str(tmp_path))  # second install is a no-op
    m = cls()
    x = torch.ones(3)
    assert torch.equal(m.forward(x, torch.tensor([[1, 28, 28], [1, 14, 28]])), x * 2)
    assert torch.equal(m.forward(x, [[1, 56, 56]], encoder_metadata=None), x * 2)
    (f,) = [p for p in os.listdir(tmp_path) if p.startswith("vit_")]
    rows = [json.loads(line) for line in open(tmp_path / f)]
    assert [(r["images"], r["patches"]) for r in rows] == [(2, 784 + 392), (1, 3136)]
    assert all(r["ms"] >= 0 and r["t"] > 0 for r in rows)


def test_unset_env_patches_nothing(monkeypatch):
    monkeypatch.delenv("VTS_VIT_TRACE", raising=False)
    assert vittrace.maybe_install() is False
