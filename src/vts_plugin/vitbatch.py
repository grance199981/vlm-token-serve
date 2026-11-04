"""Bound the ViT batch when pruning is on.

vLLM 0.32's scheduler debits the encoder compute budget by the number of
placeholder embeddings (`mm_position.get_num_embeds()`, scheduler.py:1847/1966).
With pruning the placeholder holds only the kept tokens, so one step can admit
~1/keep times more image patches than the startup profile (1 item of the
maximum size) measured, and the ViT activations can exceed the reserved memory
(CUDA OOM in the ViT MLP on large DocVQA images). Running the tower in groups of
at most `cap` patches restores the profiled peak without touching the scheduler.
Reported upstream as vllm-project/vllm#63187.
"""
import json
import os
from collections.abc import Callable, Sequence

import torch

_PIXEL_KEYS = {"max_pixels", "min_pixels", "size", "longest_edge", "shortest_edge"}


def group_by_patches(patches: Sequence[int], cap: int) -> list[tuple[int, int]]:
    """Greedy consecutive groups [lo, hi) whose patch sum is <= cap.

    An item larger than cap forms its own group; cap <= 0 means one item per group.
    """
    groups, lo, total = [], 0, 0
    for i, p in enumerate(patches):
        if i > lo and (cap <= 0 or total + p > cap):
            groups.append((lo, i))
            lo, total = i, 0
        total += p
    if len(patches):
        groups.append((lo, len(patches)))
    return groups


def vit_patch_cap(model_path: str, mm_processor_kwargs: dict | None = None) -> int:
    """Patches of the largest image the processor can emit (= what vLLM profiles).

    Returns 0 (one image per ViT call) when it cannot be determined.
    """
    if mm_processor_kwargs and _PIXEL_KEYS & set(mm_processor_kwargs):
        return 0  # pixel limits overridden per server; stay conservative
    try:
        with open(os.path.join(model_path, "preprocessor_config.json")) as f:
            pc = json.load(f)
        max_pixels = pc.get("size", {}).get("longest_edge") or pc.get("max_pixels")
        return int(max_pixels) // int(pc["patch_size"]) ** 2
    except (OSError, KeyError, TypeError, ValueError):
        return 0


def run_grouped(visual: Callable, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                cap: int, merge: int) -> tuple[tuple[torch.Tensor, ...], int]:
    """visual(pixel_values, grid_thw=...) in patch-bounded groups.

    Returns (per-image embeddings, number of tower calls). Each group's output is
    split directly (no concatenation copy of all groups).
    """
    patches = grid_thw.prod(-1).tolist()
    tokens = [p // merge // merge for p in patches]
    groups = group_by_patches(patches, cap)
    chunks = pixel_values.split(patches) if len(groups) > 1 else (pixel_values,)
    out: list[torch.Tensor] = []
    for lo, hi in groups:
        pv = pixel_values if len(groups) == 1 else torch.cat(chunks[lo:hi])
        out.extend(visual(pv, grid_thw=grid_thw[lo:hi]).split(tokens[lo:hi]))
    return tuple(out), len(groups)
