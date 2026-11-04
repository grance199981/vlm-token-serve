"""W4 ⑥ prefill length sweep (docs/plans/W4_PLAN.md §9b, exploratory): single text-only request,
random token ids (unique per run, so no prefix-cache hits), TTFT and LLM forward time per length.
Prefix caching on/off is the factor under test (on: Qwen3.5 uses mamba_cache_mode 'align').

Same engine settings as the ① stage bench (in-process, max_model_len 16384, gpu util 0.90),
plus an explicit max_num_batched_tokens of 16384 so every length is one prefill step.

Usage: python prefill_sweep.py --model <dir> --prefix-caching on|off --out <json>
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

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.model_executor.models import qwen3_5, qwen3_vl  # noqa: E402

ACC = {"llm": 0.0, "llm_calls": 0}
LENGTHS = [128, 256, 512, 822, 1024, 2048, 3180, 4096, 8192]
TOKEN_RANGE = (1000, 100000)  # ordinary vocabulary ids, below every special token


def timed(fn):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():  # CUDA-graph capture: no sync allowed
            return fn(*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        ACC["llm"] += time.perf_counter() - t0
        ACC["llm_calls"] += 1
        return out
    return wrapper


qwen3_vl.Qwen3VLForConditionalGeneration.forward = timed(qwen3_vl.Qwen3VLForConditionalGeneration.forward)
qwen3_5.Qwen3_5ForConditionalGeneration.forward = timed(qwen3_5.Qwen3_5ForConditionalGeneration.forward)


def one(llm, length, seed):
    ACC["llm"], ACC["llm_calls"] = 0.0, 0
    ids = np.random.default_rng(seed).integers(*TOKEN_RANGE, size=length).tolist()
    t0 = time.perf_counter()
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
                       use_tqdm=False)
    wall = time.perf_counter() - t0
    return {"wall": wall, "llm": ACC["llm"], "llm_calls": ACC["llm_calls"],
            "prompt_tokens": len(out[0].prompt_token_ids), "cached": out[0].num_cached_tokens or 0}


def engine_facts(llm):
    facts = {}
    cfg = getattr(llm.llm_engine, "vllm_config", None)
    if cfg is not None:
        cc = cfg.cache_config
        facts["enable_prefix_caching"] = bool(cc.enable_prefix_caching)
        facts["mamba_cache_mode"] = str(getattr(cc, "mamba_cache_mode", None))
        facts["block_size"] = cc.block_size
        facts["max_num_batched_tokens"] = cfg.scheduler_config.max_num_batched_tokens
        sizes = getattr(cfg.compilation_config, "cudagraph_capture_sizes", None) or []
        facts["max_cudagraph_capture_size"] = max(sizes) if sizes else None
        facts["cudagraph_mode"] = str(getattr(cfg.compilation_config, "cudagraph_mode", None))
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-caching", choices=["on", "off"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    ap.add_argument("--quantization", default=None, help="e.g. fp8_per_tensor (follow-up T3)")
    ap.add_argument("--quantization-config", default=None, help="JSON, e.g. an ignore list")
    a = ap.parse_args()

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=16384, max_num_batched_tokens=16384,
              gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0,
              enable_prefix_caching=(a.prefix_caching == "on"), quantization=a.quantization,
              quantization_config=json.loads(a.quantization_config) if a.quantization_config else None)
    res = {"model": os.path.basename(a.model.rstrip("/")), "prefix_caching": a.prefix_caching,
           "quantization": a.quantization, "quantization_config": a.quantization_config,
           "vllm": vllm.__version__, "plugin_loaded": "vts_plugin" in sys.modules,
           "vts_method": os.environ.get("VTS_METHOD", ""), "engine": engine_facts(llm), "lengths": {}}
    print("engine", json.dumps(res["engine"]))
    ok = res["vts_method"] == ""
    for li, length in enumerate(int(x) for x in a.lengths.split(",")):
        for w in range(a.warmup):
            one(llm, length, 10_000_000 + 1000 * li + w)  # warm-up ids differ from measured ones
        runs = [one(llm, length, 1000 * li + r) for r in range(a.runs)]
        walls = [1000 * x["wall"] for x in runs]
        row = {"ttft_ms": statistics.median(walls),
               "ttft_iqr_ms": float(np.subtract(*np.percentile(walls, [75, 25]))),
               "llm_forward_ms": 1000 * statistics.median(x["llm"] for x in runs),
               "llm_calls": statistics.median(x["llm_calls"] for x in runs),
               "prompt_tokens": statistics.median(x["prompt_tokens"] for x in runs),
               "max_cached_tokens": max(x["cached"] for x in runs)}
        row["other_ms"] = row["ttft_ms"] - row["llm_forward_ms"]
        # exact length, no cache hits; without prefix caching one prefill step. With it, Qwen3.5's
        # 'align' mode splits the prefill at block boundaries (scheduler._mamba_block_aligned_split),
        # so the number of steps is a measured quantity there, not a check.
        ok &= row["prompt_tokens"] == length and row["max_cached_tokens"] == 0
        ok &= row["llm_calls"] == 1 or (a.prefix_caching == "on" and res["engine"].get("mamba_cache_mode") == "align")
        res["lengths"][str(length)] = row
        print(length, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}))
    res["checks_ok"] = bool(ok)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    if not ok:
        print("INVALID: wrong length, a cache hit, an unexpected extra forward, or the plugin was active", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
