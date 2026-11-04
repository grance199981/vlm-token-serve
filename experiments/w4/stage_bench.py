"""W4 ① stage micro-benchmark (docs/plans/W4_PLAN.md §3): one request at a time, split TTFT into
ViT, plugin selection, LLM forward (prefill) and the rest (CPU preprocessing, scheduling).

In-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) with synchronising timers on
Qwen3_VisionTransformer.forward, vts_plugin.models.select and the top-level model forward.
The synchronisation inflates absolute times a little; use the result for proportions.
The method comes from VTS_METHOD / VTS_KEEP (unset = B0), so run one process per config.

Usage: python stage_bench.py --model <dir> --out <json> [--runs 20] [--warmup 3]
       [--shapes P,D] [--prefix-caching default|on|off]
The ⑥b diagnostic (W4_RESULTS §4) uses --shapes D with prefix caching on / off and reads the
number of LLM forward calls per prefill and the token count of each call (llm_tokens).
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.model_executor.models import qwen3_5, qwen3_vl  # noqa: E402

ACC = {"vit": 0.0, "sel": 0.0, "llm": 0.0, "vit_calls": 0, "sel_calls": 0, "llm_calls": 0}
LLM_TOKENS = []  # tokens per LLM forward call (padded batch size as the model sees it)


def timed(key, fn):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():  # CUDA-graph capture: no sync allowed
            return fn(*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        ACC[key] += time.perf_counter() - t0
        ACC[key + "_calls"] += 1
        if key == "llm":
            pos = kwargs.get("positions", args[2] if len(args) > 2 else None)  # args[0] is self
            LLM_TOKENS.append(int(pos.shape[-1]) if pos is not None else -1)
        return out
    return wrapper


qwen3_vl.Qwen3_VisionTransformer.forward = timed("vit", qwen3_vl.Qwen3_VisionTransformer.forward)
qwen3_vl.Qwen3VLForConditionalGeneration.forward = timed("llm", qwen3_vl.Qwen3VLForConditionalGeneration.forward)
qwen3_5.Qwen3_5ForConditionalGeneration.forward = timed("llm", qwen3_5.Qwen3_5ForConditionalGeneration.forward)

TEXT = ("Look at the images carefully and describe the most important objects, their colors, "
        "their positions relative to each other, and anything unusual you notice.")
SHAPES = {"P": 4, "D": 1}  # images per request, as in the W1 serving workloads


def prompt(n_images):
    pads = "<|vision_start|><|image_pad|><|vision_end|>" * n_images
    return f"<|im_start|>user\n{pads}{TEXT}<|im_end|>\n<|im_start|>assistant\n"


def images(shape_id, run, n, side):
    # same images for every config (paired), unique per run (no prefix / MM cache hits)
    rng = np.random.default_rng(1_000_000 * shape_id + run)
    return [Image.fromarray(rng.integers(0, 256, (side, side, 3), dtype=np.uint8)) for _ in range(n)]


def one(llm, shape, shape_id, run, max_tokens, side):
    for k in ACC:
        ACC[k] = 0 if k.endswith("_calls") else 0.0
    LLM_TOKENS.clear()
    req = {"prompt": prompt(SHAPES[shape]), "multi_modal_data": {"image": images(shape_id, run, SHAPES[shape], side)}}
    t0 = time.perf_counter()
    out = llm.generate([req], SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
                       use_tqdm=False)
    wall = time.perf_counter() - t0
    return {"wall": wall, "prompt_tokens": len(out[0].prompt_token_ids),
            "out_tokens": len(out[0].outputs[0].token_ids), "llm_tokens": list(LLM_TOKENS),
            "cached": out[0].num_cached_tokens or 0, **{k: ACC[k] for k in ACC}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=129)
    ap.add_argument("--side", type=int, default=896)
    ap.add_argument("--shapes", default="P,D")
    ap.add_argument("--prefix-caching", choices=["default", "on", "off"], default="default")
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (W5(b)); default: bf16 weights")
    ap.add_argument("--quantization-config", default=None, help="JSON, e.g. an ignore list (W5c)")
    ap.add_argument("--reset-prefix-cache", action="store_true",
                    help="call llm.reset_prefix_cache() after each shape (follow-up T4)")
    a = ap.parse_args()

    extra = {} if a.prefix_caching == "default" else {"enable_prefix_caching": a.prefix_caching == "on"}
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 4, "video": 0}, seed=0, quantization=a.quantization,
              quantization_config=json.loads(a.quantization_config) if a.quantization_config else None,
              **extra)
    cc = llm.llm_engine.vllm_config.cache_config
    method = os.environ.get("VTS_METHOD", "none")
    if method not in ("", "none"):  # wrap the plugin's selection call (looked up by name at call time)
        import vts_plugin.models as vm
        vm.select = timed("sel", vm.select)
    res = {"model": os.path.basename(a.model.rstrip("/")), "method": method,
           "keep": os.environ.get("VTS_KEEP", "1.0"), "vllm": vllm.__version__,
           "divprune_impl": os.environ.get("VTS_DIVPRUNE_IMPL", "torch"), "quantization": a.quantization,
           "quantization_config": a.quantization_config,
           "plugin_loaded": "vts_plugin" in sys.modules, "prefix_caching": bool(cc.enable_prefix_caching),
           "mamba_cache_mode": str(getattr(cc, "mamba_cache_mode", None)), "block_size": cc.block_size, "shapes": {}}
    for shape_id, shape in enumerate(SHAPES, start=1):
        if shape not in a.shapes.split(","):
            continue
        for w in range(a.warmup):
            one(llm, shape, shape_id + 10, w, 1, a.side)  # warm-up images differ from measured ones
        prefill = [one(llm, shape, shape_id, r, 1, a.side) for r in range(a.runs)]
        decode = [one(llm, shape, shape_id + 20, r, a.decode_tokens, a.side) for r in range(a.runs)]
        med = lambda xs, k: statistics.median(x[k] for x in xs)  # noqa: E731
        ttft = med(prefill, "wall")
        row = {
            "prompt_tokens": med(prefill, "prompt_tokens"),
            "ttft_ms": 1000 * ttft,
            "vit_ms": 1000 * med(prefill, "vit"),
            "select_ms": 1000 * med(prefill, "sel"),
            "llm_prefill_ms": 1000 * med(prefill, "llm"),
            "vit_calls": med(prefill, "vit_calls"),
            "select_calls": med(prefill, "sel_calls"),
            "llm_calls": med(prefill, "llm_calls"),
            "llm_tokens_run0": prefill[0]["llm_tokens"],
            "llm_tokens_all_same": all(x["llm_tokens"] == prefill[0]["llm_tokens"] for x in prefill),
            "max_cached_tokens": max(x["cached"] for x in prefill),
            "tpot_ms": 1000 * (med(decode, "wall") - ttft) / (a.decode_tokens - 1),
            "ttft_iqr_ms": 1000 * float(np.subtract(*np.percentile([x["wall"] for x in prefill], [75, 25]))),
        }
        row["other_ms"] = row["ttft_ms"] - row["vit_ms"] - row["select_ms"] - row["llm_prefill_ms"]
        for k in ("vit_ms", "select_ms", "llm_prefill_ms", "other_ms"):
            row[k.replace("_ms", "_share")] = row[k] / row["ttft_ms"]
        res["shapes"][shape] = row
        if a.reset_prefix_cache:
            res.setdefault("prefix_cache_reset_after", []).append([shape, bool(llm.reset_prefix_cache())])
        print(shape, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
    # the path under test must have run: ViT always, selection only for pruning methods
    pruning = method in ("random", "pool2x2", "divprune", "rate", "rate_pr")
    ok = all(r["vit_calls"] >= 1 and r["llm_prefill_ms"] > 0 for r in res["shapes"].values())
    ok &= all((r["select_calls"] >= 1) == pruning for r in res["shapes"].values())
    ok &= all(r["max_cached_tokens"] == 0 for r in res["shapes"].values())
    res["checks_ok"] = bool(ok)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    if not ok:
        print("INVALID: a timed path did not run as expected", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
