"""Start, health-check and stop one `vllm serve` process with the plugin configured.

The plugin reads its configuration from VTS_* environment variables in every vLLM
process, so each (model, method) pair gets its own server process.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .presets import Method

# variables that must not leak from the caller's shell into a server for another method
_VTS_VARS = ("VTS_METHOD", "VTS_KEEP", "VTS_SEED", "VTS_DIVPRUNE_IMPL", "VTS_TAIL",
             "VTS_VIT_TRACE", "VTS_STEP_TRACE")


@dataclass
class ServerSpec:
    root: Path                  # holds envs/vts/bin/vllm and models/<name>
    model: str                  # directory name under models/, also the served name
    method: Method
    port: int = 8041
    max_model_len: int = 16384
    limit_images: int = 4
    gpu_memory_utilization: float = 0.90
    extra_args: tuple[str, ...] = ()
    extra_env: dict = field(default_factory=dict)

    @property
    def vllm(self) -> str:
        exe = self.root / "envs" / "vts" / "bin" / "vllm"
        return str(exe) if exe.exists() else (shutil.which("vllm") or "vllm")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def build_serve_cmd(spec: ServerSpec) -> list[str]:
    return [spec.vllm, "serve", str(spec.root / "models" / spec.model),
            "--served-model-name", spec.model, "--host", "127.0.0.1", "--port", str(spec.port),
            "--dtype", "bfloat16", "--gpu-memory-utilization", f"{spec.gpu_memory_utilization:.2f}",
            "--max-model-len", str(spec.max_model_len),
            "--limit-mm-per-prompt", json.dumps({"image": spec.limit_images, "video": 0}),
            *spec.extra_args, "--seed", "0"]


def build_env(spec: ServerSpec, base: dict | None = None) -> dict:
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in _VTS_VARS}
    env.update(spec.method.env)
    env.update(spec.extra_env)
    return env


class VLLMServer:
    """Context manager: `with VLLMServer(spec, log_path) as srv: ...` waits until /health answers."""

    def __init__(self, spec: ServerSpec, log_path: Path, timeout_s: float = 1200, poll_s: float = 5):
        self.spec, self.log_path = spec, Path(log_path)
        self.timeout_s, self.poll_s = timeout_s, poll_s
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "VLLMServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(build_serve_cmd(self.spec), env=build_env(self.spec),
                                     stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self._close()
                raise RuntimeError(f"vllm serve exited with {self.proc.returncode}; see {self.log_path}")
            if self.healthy():
                return self
            time.sleep(self.poll_s)
        self._close()
        raise TimeoutError(f"vllm serve not healthy after {self.timeout_s}s; see {self.log_path}")

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.spec.base_url + "/health", timeout=5) as r:
                return r.status == 200
        except OSError:
            return False

    def log_count(self, pattern: str) -> int:
        """Lines of the server log containing `pattern` (e.g. "vts_plugin", "OutOfMemoryError")."""
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return sum(pattern in line for line in f)

    def _close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()

    def __exit__(self, *exc):
        self._close()
        return False


def gpu_busy(gpu: str) -> bool:
    """True if another process already holds memory on `gpu` (a UUID); False without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.strip() == gpu for line in out.splitlines())
