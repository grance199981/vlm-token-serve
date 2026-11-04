"""ViT grouping under pruning: CPU tests with a fake vision tower."""
import json

import pytest

torch = pytest.importorskip("torch")

from vts_plugin.vitbatch import group_by_patches, run_grouped, vit_patch_cap  # noqa: E402

MERGE = 2


class FakeVisual:
    """Row-wise stand-in for the Qwen3-VL tower: sums each run of merge**2 patch rows."""

    def __init__(self, batch_dependent=False):
        self.calls = []
        self.batch_dependent = batch_dependent

    def __call__(self, pixel_values, grid_thw):
        assert pixel_values.shape[0] == int(grid_thw.prod(-1).sum())
        self.calls.append(pixel_values.shape[0])
        out = pixel_values.view(-1, MERGE * MERGE, pixel_values.shape[1]).sum(1)
        if self.batch_dependent:  # negative control: output depends on batch composition
            out = out + grid_thw.shape[0]
        return out


def make_batch(grids, dim=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid_thw = torch.tensor(grids)
    n = int(grid_thw.prod(-1).sum())
    return torch.randn(n, dim, generator=g), grid_thw


def test_groups_respect_cap_and_order():
    patches = [16, 16, 40, 8, 8, 8, 100, 4]
    groups = group_by_patches(patches, 40)
    assert groups[0][0] == 0 and groups[-1][1] == len(patches)
    assert all(a[1] == b[0] for a, b in zip(groups, groups[1:]))  # contiguous, in order
    for lo, hi in groups:
        s = sum(patches[lo:hi])
        assert s <= 40 or hi - lo == 1  # only a single oversize item may exceed the cap
    assert (6, 7) in groups  # the 100-patch item runs alone
    assert group_by_patches(patches, 0) == [(i, i + 1) for i in range(len(patches))]
    assert group_by_patches(patches, 10**9) == [(0, len(patches))]
    assert group_by_patches([], 40) == []


def test_grouped_equals_ungrouped_and_bounds_each_call():
    # 6 images of 1x8x8 = 64 patches each (16 merged tokens); cap = 128 patches = 2 images
    pv, grid = make_batch([[1, 8, 8]] * 6)
    ref_vis, vis = FakeVisual(), FakeVisual()
    ref = ref_vis(pv, grid_thw=grid)
    parts, n_calls = run_grouped(vis, pv, grid, cap=128, merge=MERGE)
    out = torch.cat(parts)
    assert n_calls == 3 and vis.calls == [128, 128, 128]  # path really split
    assert max(ref_vis.calls) == 384 > 128  # negative control: unsplit call exceeds the cap
    assert torch.equal(out, ref)


def test_equality_check_detects_batch_dependence():
    # negative control for the equality assertion above
    pv, grid = make_batch([[1, 8, 8]] * 4, seed=1)
    ref = FakeVisual(batch_dependent=True)(pv, grid_thw=grid)
    parts, n_calls = run_grouped(FakeVisual(batch_dependent=True), pv, grid, cap=128, merge=MERGE)
    out = torch.cat(parts)
    assert n_calls == 2 and not torch.equal(out, ref)


def test_mixed_sizes_split_back_per_image():
    grids = [[1, 4, 4], [1, 16, 8], [1, 2, 2], [1, 8, 8]]
    pv, grid = make_batch(grids, seed=2)
    parts, n_calls = run_grouped(FakeVisual(), pv, grid, cap=130, merge=MERGE)
    assert n_calls == 3
    sizes = (grid.prod(-1) // MERGE // MERGE).tolist()
    assert [p.shape[0] for p in parts] == sizes
    ref_parts = FakeVisual()(pv, grid_thw=grid).split(sizes)
    assert all(torch.equal(a, b) for a, b in zip(parts, ref_parts))


def test_single_group_passthrough():
    pv, grid = make_batch([[1, 4, 4], [1, 2, 2]], seed=3)
    vis = FakeVisual()
    parts, n_calls = run_grouped(vis, pv, grid, cap=10**6, merge=MERGE)
    assert n_calls == 1 and vis.calls == [20] and [p.shape[0] for p in parts] == [4, 1]


def test_patch_cap_from_preprocessor_config(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(
        {"size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16}))
    assert vit_patch_cap(str(tmp_path)) == 65536  # = 16384 merged tokens, vLLM's profiled item
    assert vit_patch_cap(str(tmp_path), {"device": "cuda"}) == 65536
    assert vit_patch_cap(str(tmp_path), {"max_pixels": 1 << 20}) == 0
    assert vit_patch_cap(str(tmp_path / "missing")) == 0
