"""Run-time configuration of the VLM-TokenServe plugin.

Everything is read from environment variables so that the API-server process
(which runs the multimodal processor and decides placeholder counts) and the
EngineCore process (which runs the vision encoder and selects tokens) see exactly
the same configuration.

VTS_METHOD  none | keep_all | random | pool2x2 | divprune | rate | rate_pr | rate_orig
            | downsample                                              (default none)
            none       plugin inactive (stock vLLM behaviour)
            keep_all   pruning machinery on, every token kept (overhead control)
            random     fixed random subset per (seed, token count)
            pool2x2    average 2x2 blocks of the merged token grid
            divprune   greedy max-min diversity (DivPrune)
            rate       RATE from vLLM PR #60963 with a descending projection sort
            rate_pr    the PR as written at 06158cbe (ascending sort; keeps a prefix)
            rate_orig  the PR at 45333ba (the author's fix), copied line for line
            downsample resize every image by sqrt(VTS_KEEP) per side before the HF
                       processor (no token pruning; the pruning machinery stays off)
VTS_KEEP    fraction of visual tokens kept, (0, 1]                    (default 1.0)
VTS_SEED    seed for random selection / RATE projections              (default 0)
VTS_DIVPRUNE_IMPL  torch | triton   divprune only: the PyTorch greedy loop or the
            single Triton kernel (divprune_triton.py; large images fall back to
            torch)                                                    (default torch)
VTS_TAIL    original | compact                                        (default original)
            original: text after an image keeps its unpruned M-RoPE positions
            compact : text continues right after the max retained position
                      (EVS / RATE behaviour; unsafe with prefix caching)
VTS_DEBUG_POS_SHIFT      TEST ONLY: shift text after every image by this many
                         positions (deliberate position error, negative control)
VTS_DEBUG_POS_TRANSPOSE  TEST ONLY: 1 = swap row/col of every retained image token
"""
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

METHODS = ("none", "keep_all", "random", "pool2x2", "divprune", "rate", "rate_pr", "rate_orig",
           "downsample")
PRUNING_METHODS = ("keep_all", "random", "pool2x2", "divprune", "rate", "rate_pr", "rate_orig")
TAILS = ("original", "compact")
DIVPRUNE_IMPLS = ("torch", "triton")


@dataclass(frozen=True)
class VTSConfig:
    method: str = "none"
    keep: float = 1.0
    seed: int = 0
    tail: str = "original"
    debug_pos_shift: int = 0
    debug_pos_transpose: bool = False
    divprune_impl: str = "torch"

    @property
    def enabled(self) -> bool:
        """The plugin registers its model classes."""
        return self.method != "none"

    @property
    def prunes(self) -> bool:
        """Image tokens are selected after the vision encoder (placeholders shrink)."""
        return self.method in PRUNING_METHODS


def parse_config(env: Mapping[str, str] | None = None) -> VTSConfig:
    """Parse and validate the VTS_* variables of `env` (default: os.environ)."""
    env = os.environ if env is None else env
    method = env.get("VTS_METHOD", "none").strip().lower()
    keep = float(env.get("VTS_KEEP", "1.0"))
    seed = int(env.get("VTS_SEED", "0"))
    tail = env.get("VTS_TAIL", "original").strip().lower()
    if method not in METHODS:
        raise ValueError(f"VTS_METHOD={method!r} not in {METHODS}")
    if tail not in TAILS:
        raise ValueError(f"VTS_TAIL={tail!r} not in {TAILS}")
    if not 0.0 < keep <= 1.0:
        raise ValueError(f"VTS_KEEP={keep} must be in (0, 1]")
    opts = {}
    impl = env.get("VTS_DIVPRUNE_IMPL")
    if impl is not None:
        if method != "divprune":
            raise ValueError("VTS_DIVPRUNE_IMPL only applies to VTS_METHOD=divprune")
        impl = impl.strip().lower()
        if impl not in DIVPRUNE_IMPLS:
            raise ValueError(f"VTS_DIVPRUNE_IMPL={impl!r} not in {DIVPRUNE_IMPLS}")
        opts["divprune_impl"] = impl
    return VTSConfig(method=method, keep=keep, seed=seed, tail=tail,
                     debug_pos_shift=int(env.get("VTS_DEBUG_POS_SHIFT", "0")),
                     debug_pos_transpose=env.get("VTS_DEBUG_POS_TRANSPOSE", "0") == "1",
                     **opts)


@lru_cache(maxsize=1)
def get_config() -> VTSConfig:
    """Process-wide configuration, parsed once."""
    return parse_config()


def keep_count(cfg: VTSConfig, num_tokens: int, grid_h: int, grid_w: int) -> int:
    """Number of visual tokens an image keeps.

    Must give the same answer in the processor (placeholder count, API-server
    process) and in the model (selected embeddings, EngineCore process).
    """
    assert num_tokens == grid_h * grid_w, (num_tokens, grid_h, grid_w)
    if cfg.method in ("none", "keep_all", "downsample"):
        return num_tokens
    if cfg.method == "pool2x2":
        return math.ceil(grid_h / 2) * math.ceil(grid_w / 2)
    return max(1, int(num_tokens * cfg.keep))
