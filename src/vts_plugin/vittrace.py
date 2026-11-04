"""Opt-in ViT call trace (used for the ViT time-share and batch-composition measurements).

VTS_VIT_TRACE=<dir> wraps Qwen3_VisionTransformer.forward in every vLLM process that loads the
plugin and appends one JSON line per call to <dir>/vit_<pid>.jsonl:
  {"t": wall time at the end of the call, "ms": duration, "images": images in the call,
   "patches": sum of t*h*w}
The call is bracketed by torch.cuda.synchronize(), so tracing costs a little throughput; use the
traced runs for time shares and batch composition, not for headline throughput. Calls made
during CUDA-graph capture are passed through untimed. Unset = nothing is patched.
"""
import json
import os
import time


def _grid_stats(grid_thw):
    rows = grid_thw.tolist() if hasattr(grid_thw, "tolist") else list(grid_thw)
    return len(rows), int(sum(int(t) * int(h) * int(w) for t, h, w in rows))


def install(cls, out_dir):
    """Wrap cls.forward(self, x, grid_thw, ...) once; returns True if it patched."""
    if getattr(cls, "_vts_traced", False):
        return False
    import torch

    os.makedirs(out_dir, exist_ok=True)
    orig = cls.forward
    path = os.path.join(out_dir, f"vit_{os.getpid()}.jsonl")
    cuda = torch.cuda.is_available()

    def forward(self, x, grid_thw, *args, **kwargs):
        if cuda and torch.cuda.is_current_stream_capturing():
            return orig(self, x, grid_thw, *args, **kwargs)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(self, x, grid_thw, *args, **kwargs)
        if cuda:
            torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        images, patches = _grid_stats(grid_thw)
        with open(path, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 4), "ms": round(ms, 3), "images": images,
                                "patches": patches}) + "\n")
        return out

    cls.forward = forward
    cls._vts_traced = True
    return True


def maybe_install():
    out_dir = os.environ.get("VTS_VIT_TRACE")
    if not out_dir:
        return False
    from vllm.model_executor.models import qwen3_vl  # Qwen3.5 reuses this ViT class

    return install(qwen3_vl.Qwen3_VisionTransformer, out_dir)
