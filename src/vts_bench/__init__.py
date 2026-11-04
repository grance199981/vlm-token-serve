"""vts-bench: one command line for the serving and accuracy measurements of VLM-TokenServe.

    vts-bench serve      start `vllm serve` per (model, method), drive a workload preset
                         (P / D / K1) with `vllm bench serve`, save one JSON per point
    vts-bench accuracy   start `vllm serve`, run lmms-eval tasks against it
    vts-bench summarize  throughput gains vs the uncompressed baseline, or paired
                         accuracy deltas with bootstrap CIs

The measurement protocol (workload shapes, seeds, warm-up, server flags) is the one used
for the published numbers (experiments/w1, w3, w4); see presets.py.
"""
__version__ = "0.2.0"
