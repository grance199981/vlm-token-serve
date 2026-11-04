"""Sample vLLM's Prometheus /metrics once per second during a benchmark point.

Columns: unix time, requests running, requests waiting, KV-cache usage (fraction),
cumulative preemptions. Same parsing as experiments/w4/metrics_sampler.py.
"""
import re
import threading
import time
import urllib.request
from pathlib import Path

KEYS = {"running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting",
        "kv_usage": "vllm:kv_cache_usage_perc", "preemptions": "vllm:num_preemptions_total"}


def metric_value(text: str, name: str) -> float:
    """Sum of all samples of `name` (over label sets); NaN if absent."""
    total, seen = 0.0, False
    for m in re.finditer(r"^" + re.escape(name) + r"(\{[^}]*\})?\s+([0-9.eE+-]+)$", text, re.M):
        total += float(m.group(2))
        seen = True
    return total if seen else float("nan")


class MetricsSampler(threading.Thread):
    """`with MetricsSampler(url, csv_path): run_point()` writes one CSV row per interval."""

    def __init__(self, url: str, out: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.url, self.out, self.interval = url, Path(out), interval
        self._halt = threading.Event()

    def run(self):
        fails = 0
        with open(self.out, "w") as f:
            f.write("t," + ",".join(KEYS) + "\n")
            while not self._halt.is_set():
                try:
                    with urllib.request.urlopen(self.url, timeout=5) as r:
                        text = r.read().decode()
                    fails = 0
                except OSError:
                    fails += 1
                    if fails > 10:
                        break
                    self._halt.wait(self.interval)
                    continue
                vals = ",".join(f"{metric_value(text, k):g}" for k in KEYS.values())
                f.write(f"{time.time():.3f},{vals}\n")
                f.flush()
                self._halt.wait(self.interval)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self._halt.set()
        self.join(timeout=10)
        return False
