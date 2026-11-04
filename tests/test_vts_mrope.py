"""T4: M-RoPE placeholder + recompute for pruned images, every chunk boundary.

Expected positions are built directly: text consecutive; each image's retained
tokens sit at base + (0, row, col) of the FULL grid; text after an image starts
at base + max(gh, gw) ("original" tail) or right after the max retained
position ("compact" tail).
"""
import itertools

import pytest

torch = pytest.importorskip("torch")

from vts_plugin.mrope import initial_positions, recompute_image_positions  # noqa: E402

VS, IMG, VE, TXT = 900, 901, 902, 1


def build(images, text_before=3, text_between=2, text_after=4):
    """images: list of (gh, gw, kept_raster_indices). Returns ids, image specs,
    per-image retained (t,h,w) rows and per-token expected positions."""
    ids, specs, kept_pos = [], [], []
    for i, (gh, gw, kept) in enumerate(images):
        ids += [TXT] * (text_before if i == 0 else text_between) + [VS]
        specs.append((len(ids), len(kept), gh, gw))
        ids += [IMG] * len(kept) + [VE]
        rows = torch.tensor([[0, r // gw, r % gw] for r in kept]).T  # (3, k)
        kept_pos.append(rows)
    ids += [TXT] * text_after
    return torch.tensor(ids), specs, kept_pos


def expected(ids, specs, kept_pos, tail):
    n = len(ids)
    pos = torch.zeros(3, n, dtype=torch.long)
    nxt, st = 0, 0
    for (offset, length, gh, gw), rows in zip(specs, kept_pos):
        for j in range(st, offset):
            pos[:, j] = nxt
            nxt += 1
        base = nxt
        pos[:, offset:offset + length] = rows + base
        nxt = base + (max(gh, gw) if tail == "original" else int(rows.max()) + 1)
        st = offset + length
    for j in range(st, n):
        pos[:, j] = nxt
        nxt += 1
    return pos


def run_chunked(ids, specs, kept_pos, boundaries, tail):
    pos, _ = initial_positions(len(ids), specs)
    media_idx = ids.eq(IMG).nonzero(as_tuple=True)[0].tolist()
    # mm channels per media token in prompt order: t, h, w, gw, gh
    chans = []
    for (offset, length, gh, gw), rows in zip(specs, kept_pos):
        for j in range(length):
            chans.append([*rows[:, j].tolist(), gw, gh])
    chans = torch.tensor(chans).T
    starts = [0] + list(boundaries)
    ends = list(boundaries) + [len(ids)]
    for s, e in zip(starts, ends):
        sel = [k for k, m in enumerate(media_idx) if s <= m < e]
        if not sel:
            continue
        # split into per-image pieces, as vLLM passes one tensor per media item
        pieces, cur = [], [sel[0]]
        for a, b in zip(sel, sel[1:]):
            if media_idx[b] == media_idx[a] + 1:
                cur.append(b)
            else:
                pieces.append(cur)
                cur = [b]
        pieces.append(cur)
        chunk = [chans[:, p] for p in pieces]
        pos, delta = recompute_image_positions(ids, chunk, pos, s, VS, IMG, tail)
    return pos, delta


# retained sets chosen so that compact and original tails differ for images 1-2
IMAGES = [(3, 4, [0, 2, 5]), (2, 5, [1, 3, 8]), (4, 2, [6])]


@pytest.mark.parametrize("tail", ["original", "compact"])
def test_every_single_boundary(tail):
    ids, specs, kept = build(IMAGES)
    want = expected(ids, specs, kept, tail)
    for b in range(1, len(ids)):
        got, delta = run_chunked(ids, specs, kept, [b], tail)
        assert torch.equal(got, want), f"boundary {b}"
        assert delta == int(want.max() + 1 - len(ids))


@pytest.mark.parametrize("tail", ["original", "compact"])
def test_many_boundaries(tail):
    ids, specs, kept = build(IMAGES)
    want = expected(ids, specs, kept, tail)
    for bs in itertools.combinations(range(1, len(ids)), 3):
        got, _ = run_chunked(ids, specs, kept, list(bs), tail)
        assert torch.equal(got, want), f"boundaries {bs}"


def test_keep_all_equals_vanilla_layout():
    # full grids kept: identical to vLLM's unpruned M-RoPE layout
    imgs = [(3, 4, list(range(12))), (2, 5, list(range(10)))]
    ids, specs, kept = build(imgs)
    pos, _ = initial_positions(len(ids), specs)
    assert torch.equal(pos, expected(ids, specs, kept, "original"))
    assert torch.equal(expected(ids, specs, kept, "original"), expected(ids, specs, kept, "compact"))


def test_original_tail_is_final_before_recompute():
    # text positions after a pruned image are already final at preprocessing time
    ids, specs, kept = build(IMAGES)
    pos0, _ = initial_positions(len(ids), specs)
    want = expected(ids, specs, kept, "original")
    text = ~ids.eq(IMG)
    assert torch.equal(pos0[:, text], want[:, text])


def test_negative_control_wrong_tail_detected():
    ids, specs, kept = build(IMAGES)
    got, _ = run_chunked(ids, specs, kept, [7], "compact")
    assert not torch.equal(got, expected(ids, specs, kept, "original"))
