"""M-RoPE positions for pruned images (image-only, no vLLM imports).

Placeholder positions at preprocessing time (`initial_positions`):
  text      : consecutive, as in vLLM
  image     : the first `length` entries of the full (1, gh, gw) grid, as a
              placeholder; overwritten by `recompute_image_positions`
  after it  : starts at base + max(gh, gw), i.e. exactly where it would start
              for the unpruned image ("original" tail). This makes every text
              position final at preprocessing time, so prefix-cache hits that
              skip the encoder cannot leave stale positions behind.

`recompute_image_positions` writes each retained token's original grid
position (carried in its appended channels) and, for the "compact" tail
(EVS / RATE behaviour), shifts everything after a completed image so that text
continues right after the max retained position.

Unlike vLLM 0.32's evs.recompute_mrope_positions it locates the image a chunk
belongs to by the global index of its first media token, so a prefill chunk
boundary that falls between <vision_start> and the first image token is handled
(the vLLM routine then binds the chunk to the *next* image or raises).
"""
import numpy as np
import torch


def initial_positions(num_tokens, images, debug_shift=0):
    """images: list of (offset, length, grid_h, grid_w), sorted by offset.
    Returns (positions int64 tensor (3, num_tokens), delta)."""
    parts = []
    st = 0          # next sequence index to fill
    nxt = 0         # next position value
    for offset, length, gh, gw in images:
        text_len = offset - st
        parts.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + nxt)
        base = nxt + text_len
        full = np.indices((1, gh, gw)).reshape(3, -1)
        parts.append(full[:, :length] + base)
        nxt = base + max(gh, gw) + debug_shift  # debug_shift != 0 only in tests
        st = offset + length
    if st < num_tokens:
        text_len = num_tokens - st
        parts.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + nxt)
    pos = np.concatenate(parts, axis=1).reshape(3, -1) if parts else np.zeros((3, 0), np.int64)
    delta = int(pos.max() + 1 - num_tokens) if pos.size else -num_tokens
    return torch.from_numpy(np.ascontiguousarray(pos)).long(), delta


def recompute_image_positions(input_ids, chunk_positions, positions, num_computed,
                              vision_start_id, image_id, tail="original"):
    """input_ids (N,), positions (3, N) long; chunk_positions: list of (C, L) long
    tensors (rows t, h, w, ...) for consecutive media tokens starting at the first
    media token at or after `num_computed`. Returns (positions, delta)."""
    positions = positions.clone()
    n = input_ids.numel()
    media_mask = input_ids.eq(image_id)
    media_idx = media_mask.nonzero(as_tuple=True)[0]
    seen = int(media_mask[:num_computed].sum().item())
    vs_idx = input_ids.eq(vision_start_id).nonzero(as_tuple=True)[0]
    for mm_pos in chunk_positions:
        length = mm_pos.shape[1]
        if length == 0:
            continue
        idxs = media_idx[seen:seen + length]
        assert idxs.numel() == length, "more embeddings than media placeholders"
        first = int(idxs[0])
        vs = int(vs_idx[vs_idx < first][-1])
        # image tokens are contiguous right after <vision_start>
        img_len = int(media_mask[vs + 1:].long().cumprod(0).sum().item())
        img_end = vs + img_len  # inclusive index of the image's last token
        assert int(idxs[-1]) <= img_end, "chunk spans two images"
        base = positions[0, vs] + 1
        positions[:, idxs] = mm_pos[0:3].to(positions.dtype) + base
        if tail == "compact" and int(idxs[-1]) == img_end:
            old_next = int(positions[0, img_end + 1]) if img_end + 1 < n else None
            new_next = int(positions[:, vs + 1:img_end + 1].max()) + 1
            if old_next is not None and old_next != new_next:
                positions[:, img_end + 1:] += new_next - old_next
        seen += length
    delta = int(positions.max().item() + 1 - n)
    return positions, delta
