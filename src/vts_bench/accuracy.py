"""`vts-bench accuracy`: one server per (model, method), lmms-eval tasks against it.

Server flags and lmms-eval arguments are those of experiments/w1/run_accuracy.sh and
experiments/w3/run_w3.sh, so a run here pairs per doc_id with the published B0 runs.
Tasks are the local `vts_*` task definitions (experiments/w1/make_local_tasks.py writes them
to $VTS_ROOT/tasks_local).
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .presets import (ACCURACY_CONCURRENCY, ACCURACY_LIMIT_IMAGES, ACCURACY_MAX_MODEL_LEN,
                      ACCURACY_SERVE_ARGS, ACCURACY_TASK_LIMITS, Method)
from .server import ServerSpec, VLLMServer, gpu_busy


@dataclass
class AccuracyRun:
    root: Path
    out: Path
    models: list[str]
    methods: list[Method]
    tasks: list[str]
    limit: int | None = None     # cap every task (smoke runs); None = the published limits
    port: int = 8061
    gpu: str | None = None


def lmms_eval_cmd(run: AccuracyRun, model: str, task: str, out_dir: Path) -> list[str]:
    exe = run.root / "envs" / "vts" / "bin" / "lmms-eval"
    model_args = (f"model={model},base_url=http://127.0.0.1:{run.port}/v1,api_key=EMPTY,"
                  f"num_concurrent={ACCURACY_CONCURRENCY},timeout=600,max_size_in_mb=64,"
                  "enable_thinking_kwarg=False,httpx_trust_env=False")
    cmd = [str(exe) if exe.exists() else "lmms-eval", "eval", "--model", "openai",
           "--model_args", model_args, "--tasks", task,
           "--include_path", str(run.root / "tasks_local"),
           "--batch_size", "1", "--seed", "0", "--log_samples", "--output_path", str(out_dir)]
    limit = run.limit if run.limit is not None else ACCURACY_TASK_LIMITS.get(task)
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def run_accuracy(run: AccuracyRun, log=print) -> int:
    failures = 0
    for model in run.models:
        for method in run.methods:
            if run.gpu and gpu_busy(run.gpu):
                raise RuntimeError(f"GPU {run.gpu} is in use by another process; not starting")
            cfg_dir = run.out / model / method.label
            spec = ServerSpec(run.root, model, method, port=run.port, max_model_len=ACCURACY_MAX_MODEL_LEN,
                              limit_images=ACCURACY_LIMIT_IMAGES, extra_args=ACCURACY_SERVE_ARGS)
            failed_tasks = 0
            with VLLMServer(spec, cfg_dir / "serve.log") as srv:
                for task in run.tasks:
                    with open(cfg_dir / f"lmms_eval_{task}.log", "w") as f:
                        rc = subprocess.run(lmms_eval_cmd(run, model, task, cfg_dir / "lmms_eval"),
                                            stdout=f, stderr=subprocess.STDOUT,
                                            env={**os.environ, "OPENAI_API_KEY": "EMPTY"}).returncode
                    # lmms-eval scores a request that failed 5 times as wrong instead of erroring
                    lost = (cfg_dir / f"lmms_eval_{task}.log").read_text(encoding="utf-8", errors="replace").count(
                        "All 5 attempts failed")
                    failed_tasks += rc != 0 or lost > 0
                    log(f"{model}/{method.label}/{task}: rc={rc} failed_requests={lost}")
                bad = srv.log_count("OutOfMemoryError") + srv.log_count("EngineDeadError")
            valid = failed_tasks == 0 and bad == 0
            (cfg_dir / "valid.txt").write_text("yes\n" if valid else "NO\n")
            failures += not valid
    return failures
