"""Repro for vllm-project/vllm#63187: does vLLM's own video pruning
let one scheduler step send more patches to the ViT than the startup profile?

Stock vLLM only: VLLM_PLUGINS="" (vts_plugin not loaded), in-process engine so a
class-level hook on the Qwen3-VL vision tower sees every encoder call.

D1: --pruning-rate 0.75 (EVS). D2 (control): --pruning-rate 0. Same synthetic
videos, same seed. Hypothesis: D1 has calls with more patches than the largest
profile call; D2 never does. Writes one JSON per run.

Usage: python encoder_budget_repro.py --model <dir> --out <json> [--pruning-rate 0.75]
       [--max-num-batched-tokens 16384] [--num-requests 16] [--frames 32]
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ["VLLM_PLUGINS"] = ""  # stock vLLM: no general plugins (vts_plugin stays out)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # engine in this process -> hook sees ViT calls

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.model_executor.models import qwen3_vl  # noqa: E402

CALLS = []  # (phase, n_items, n_patches)
PHASE = {"name": "profile"}
_orig_forward = qwen3_vl.Qwen3_VisionTransformer.forward


def _hooked_forward(self, x, grid_thw, *args, **kwargs):
    g = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
    CALLS.append((PHASE["name"], len(g), int(sum(t * h * w for t, h, w in g))))
    return _orig_forward(self, x, grid_thw, *args, **kwargs)


qwen3_vl.Qwen3_VisionTransformer.forward = _hooked_forward

PROMPT = ("<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
          "Describe this video in one sentence.<|im_end|>\n<|im_start|>assistant\n")


def make_video(rng, frames, height, width):
    video = rng.integers(0, 256, size=(frames, height, width, 3), dtype=np.uint8)
    meta = {"fps": 2.0, "duration": frames / 2.0, "total_num_frames": frames,
            "frames_indices": list(range(frames)), "video_backend": "opencv",
            "do_sample_frames": False}
    return video, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pruning-rate", type=float, default=0.75)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--num-requests", type=int, default=16)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    kw = dict(model=a.model, dtype="bfloat16", max_model_len=32768, max_num_seqs=64,
              gpu_memory_utilization=a.gpu_memory_utilization, seed=a.seed,
              max_num_batched_tokens=a.max_num_batched_tokens,
              limit_mm_per_prompt={"image": 0, "video": 1})
    if a.pruning_rate > 0:
        kw["video_pruning_rate"] = a.pruning_rate
    result = {"config": vars(a), "vllm": vllm.__version__, "engine_kwargs": {k: str(v) for k, v in kw.items()}}

    t0 = time.time()
    llm = LLM(**kw)
    result["init_s"] = round(time.time() - t0, 1)
    rng = np.random.default_rng(a.seed)
    reqs = [{"prompt": PROMPT, "multi_modal_data": {"video": make_video(rng, a.frames, a.height, a.width)}}
            for _ in range(a.num_requests)]
    PHASE["name"] = "generate"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        outs = llm.generate(reqs, SamplingParams(max_tokens=8, temperature=0.0))
        result["error"] = None
        result["n_outputs"] = len(outs)
        result["prompt_tokens"] = [len(o.prompt_token_ids) for o in outs]
    except Exception as e:  # an OOM here is a result, not a crash of the experiment
        result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        result["traceback_tail"] = traceback.format_exc()[-2000:]
    result["generate_s"] = round(time.time() - t0, 1)
    result["peak_allocated_gib_generate"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    prof = [c for c in CALLS if c[0] == "profile"]
    gen = [c for c in CALLS if c[0] == "generate"]
    cap = max((c[2] for c in prof), default=0)
    result.update({
        "vts_plugin_loaded": "vts_plugin" in sys.modules,
        "profile_calls": prof,
        "profile_max_patches": cap,
        "generate_calls": [c[1:] for c in gen],
        "generate_max_patches": max((c[2] for c in gen), default=0),
        "generate_calls_over_profile": sum(c[2] > cap for c in gen),
        "generate_max_items_per_call": max((c[1] for c in gen), default=0),
    })
    result["max_ratio_vs_profile"] = round(result["generate_max_patches"] / cap, 3) if cap else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ("error", "vts_plugin_loaded", "profile_max_patches",
                                              "generate_max_patches", "generate_calls_over_profile",
                                              "generate_max_items_per_call", "max_ratio_vs_profile",
                                              "peak_allocated_gib_generate")}))
    # the hook must have seen the profile run and real calls, otherwise the test is idle
    if not prof or (result["error"] is None and not gen) or result["vts_plugin_loaded"]:
        print("INVALID: hook idle or plugin loaded", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
