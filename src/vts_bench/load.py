"""`vts-bench serve`: drive a workload preset against one server per (model, method).

For every repetition, method and model: start `vllm serve` with the plugin configured,
run a discarded warm-up with its own seed, then one `vllm bench serve` point per
concurrency. Results: <out>/results/<model>_<preset>_<method>_k<keep>_b<budget>_c<conc>_rep<r>.json
(the naming that summarize.load_serve_points and experiments/w4/serve_summary.py read).
"""
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .metrics import MetricsSampler
from .presets import Method, Workload
from .server import ServerSpec, VLLMServer, gpu_busy

WARMUP_SEED = 999999


@dataclass
class LoadRun:
    root: Path
    out: Path
    workload: Workload
    models: list[str]
    methods: list[Method]
    concurrencies: list[int]
    reps: int = 3
    budgets: tuple[str, ...] = ("default",)  # --max-num-batched-tokens values; "default" = vLLM's
    kv_bytes: int | None = None              # --kv-cache-memory-bytes (required for K1)
    port: int = 8041
    gpu: str | None = None                   # refuse to start if another process holds this GPU


def bench_cmd(run: LoadRun, model: str, num_prompts: int, conc: int, seed: int,
              output_len: int | None = None, result: Path | None = None) -> list[str]:
    w = run.workload
    vllm = ServerSpec(run.root, model, Method("none", "none", 1.0)).vllm
    cmd = [vllm, "bench", "serve", "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
           "--base-url", f"http://127.0.0.1:{run.port}", "--model", model,
           "--tokenizer", str(run.root / "models" / model),
           "--dataset-name", "random-mm", "--random-input-len", str(w.input_len),
           "--random-output-len", str(w.output_len if output_len is None else output_len),
           "--random-range-ratio", "0",
           "--random-mm-base-items-per-request", str(w.images), "--random-mm-num-mm-items-range-ratio", "0",
           "--random-mm-limit-mm-per-prompt", json.dumps({"image": w.images, "video": 0}),
           "--random-mm-bucket-config", f"{{({w.image_side}, {w.image_side}, 1): 1.0}}",
           "--ignore-eos", "--temperature", "0",
           "--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": False}}),
           "--num-prompts", str(num_prompts), "--max-concurrency", str(conc),
           "--request-rate", "inf", "--seed", str(seed)]
    if result is not None:
        cmd += ["--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
                "--save-result", "--result-dir", str(result.parent), "--result-filename", result.name]
    return cmd


def server_spec(run: LoadRun, model: str, method: Method, budget: str) -> ServerSpec:
    w = run.workload
    extra = []
    if budget != "default":
        extra += ["--max-num-batched-tokens", budget]
    if w.name == "K1":
        if run.kv_bytes is None:
            raise ValueError("preset K1 needs --kv-bytes (the same KV byte budget for every model)")
        extra += ["--kv-cache-memory-bytes", str(run.kv_bytes)]
    if w.max_num_seqs is not None:
        extra += ["--max-num-seqs", str(w.max_num_seqs)]
    return ServerSpec(run.root, model, method, port=run.port, max_model_len=w.max_model_len,
                      limit_images=w.limit_images, extra_args=tuple(extra))


def point_name(model: str, w: Workload, method: Method, budget: str, conc: int, rep: int) -> str:
    return f"{model}_{w.name}_{method.label}_b{budget}_c{conc}_rep{rep}"


def run_load(run: LoadRun, log=print) -> int:
    """Run every point; returns the number of failed points."""
    results = run.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    w, failures = run.workload, 0
    for rep in range(1, run.reps + 1):
        for bi, budget in enumerate(run.budgets):
            for method in run.methods:
                for model in run.models:
                    if run.gpu and gpu_busy(run.gpu):
                        raise RuntimeError(f"GPU {run.gpu} is in use by another process; not starting")
                    srv_name = f"{model}_{method.label}_b{budget}_rep{rep}"
                    log(f"{srv_name}: starting vllm serve")
                    spec = server_spec(run, model, method, budget)
                    with VLLMServer(spec, run.out / f"serve_{srv_name}.log") as srv:
                        log(f"{srv_name}: healthy, plugin log lines {srv.log_count('vts_plugin')}")
                        warm = bench_cmd(run, model, *w.warmup, WARMUP_SEED, output_len=w.warmup_output_len)
                        _run(warm, run.out / f"warmup_{srv_name}.log")
                        for conc in run.concurrencies:
                            name = point_name(model, w, method, budget, conc, rep)
                            cmd = bench_cmd(run, model, w.num_prompts(conc), conc, w.seed(rep, conc, bi),
                                            result=results / f"{name}.json")
                            if w.metrics:
                                with MetricsSampler(spec.base_url + "/metrics", results / f"{name}.metrics.csv"):
                                    rc = _run(cmd, results / f"{name}.log")
                            else:
                                rc = _run(cmd, results / f"{name}.log")
                            failures += rc != 0
                            log(f"{name}: rc={rc}")
                        n_oom = srv.log_count("OutOfMemoryError") + srv.log_count("EngineDeadError")
                        if n_oom:
                            log(f"{srv_name}: {n_oom} OOM / engine-dead lines in the server log")
                            failures += 1
    return failures


def _run(cmd: list[str], log_path: Path) -> int:
    with open(log_path, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    sys.exit("use the `vts-bench serve` command")
