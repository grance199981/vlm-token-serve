"""Workload presets and server flags, copied from the experiment runners that produced the
published numbers (experiments/w4/run_w4_serving.sh, experiments/w4/run_k1.sh,
experiments/w3/run_w3.sh). Every preset is closed-loop: a fixed concurrency of
synthetic random-mm requests (`vllm bench serve --dataset-name random-mm`) with
--ignore-eos, so output lengths are exact.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Workload:
    name: str
    images: int                 # images per request
    image_side: int             # square images, pixels per side
    input_len: int              # text tokens per request
    output_len: int             # generated tokens per request (ignore-eos)
    seed_base: int              # per-point seed = seed(rep, conc, budget_index)
    workload_id: int            # distinguishes P / D seeds that share seed_base
    serve_args: tuple[str, ...] = ()
    limit_images: int = 4       # --limit-mm-per-prompt image count on the server
    max_model_len: int = 16384
    warmup: tuple[int, int] = (16, 8)  # (prompts, concurrency) of the discarded warm-up
    warmup_output_len: int | None = None  # None = output_len
    max_num_seqs: int | None = None
    metrics: bool = False       # sample /metrics during each point (running / KV usage)

    def num_prompts(self, conc: int, min_prompts: int = 32) -> int:
        """Prompts per point: enough for the concurrency to saturate, capped."""
        if self.name == "K1":
            return max(conc * 4, 256)
        return min(max(conc * 8, min_prompts), 256)

    def seed(self, rep: int, conc: int, budget_index: int = 0) -> int:
        """Unique per point, shared across models / methods (paired comparisons)."""
        if self.name == "K1":
            return self.seed_base + rep * 10000 + conc
        return self.seed_base + rep * 10000 + budget_index * 1000 + conc * 10 + self.workload_id


WORKLOADS = {
    # prefill-bound: 4 images of 896x896, 32 output tokens; swept over --max-num-batched-tokens
    "P": Workload("P", images=4, image_side=896, input_len=64, output_len=32,
                  seed_base=400000, workload_id=1),
    # decode-bound: 1 image, 512 output tokens
    "D": Workload("D", images=1, image_side=896, input_len=64, output_len=512,
                  seed_base=400000, workload_id=2),
    # memory-bound: same KV byte budget for both models (--kv-cache-memory-bytes), 2 images
    "K1": Workload("K1", images=2, image_side=896, input_len=64, output_len=256,
                   seed_base=300000, workload_id=0, limit_images=2, max_model_len=4096,
                   warmup_output_len=32, max_num_seqs=128, metrics=True),
}

# Accuracy runs: the W1 / W3 server flags (model-default image size, one image per prompt).
ACCURACY_MAX_MODEL_LEN = 32768
ACCURACY_SERVE_ARGS = ("--max-num-seqs", "64")
ACCURACY_LIMIT_IMAGES = 1
ACCURACY_CONCURRENCY = 32
ACCURACY_TASK_LIMITS = {"vts_textvqa_val": 1500, "vts_docvqa_val": 1000,
                        "vts_mmbench_en_dev_static": 1500}

# Method tokens accepted on the command line (in addition to config.METHODS):
#   divprune_triton = divprune with VTS_DIVPRUNE_IMPL=triton
METHOD_ALIASES = {"divprune_triton": ("divprune", {"VTS_DIVPRUNE_IMPL": "triton"})}


@dataclass(frozen=True)
class Method:
    token: str                  # as given on the command line, used in result names
    vts_method: str             # VTS_METHOD value ("none" = plugin off)
    keep: float
    extra_env: dict = field(default_factory=dict)

    @property
    def env(self) -> dict:
        if self.vts_method == "none":
            return {}
        return {"VTS_METHOD": self.vts_method, "VTS_KEEP": str(self.keep), "VTS_SEED": "0",
                **self.extra_env}

    @property
    def label(self) -> str:
        return f"{self.token}_k{1.0 if self.vts_method == 'none' else self.keep}"


def parse_method(token: str, keep: float) -> Method:
    base, extra = METHOD_ALIASES.get(token, (token, {}))
    return Method(token=token, vts_method=base, keep=1.0 if base == "none" else keep,
                  extra_env=dict(extra))
