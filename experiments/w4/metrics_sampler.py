"""Sample vLLM's Prometheus /metrics every --interval seconds into a CSV (W4 K1).

Columns: unix time, num_requests_running, num_requests_waiting, kv_cache_usage_perc,
num_preemptions_total. Also saves the first full /metrics dump (cache_config_info carries
num_gpu_blocks). Stops when --stop-file exists or the server stops answering.

Usage: python metrics_sampler.py --url http://127.0.0.1:8071/metrics --out x.csv --stop-file x.stop
"""
import argparse
import os
import re
import time
import urllib.request

KEYS = {"running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting",
        "kv_usage": "vllm:kv_cache_usage_perc", "preemptions": "vllm:num_preemptions_total"}


def scrape(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def value(text, name):
    total, seen = 0.0, False
    for m in re.finditer(r"^" + re.escape(name) + r"(\{[^}]*\})?\s+([0-9.eE+-]+)$", text, re.M):
        total += float(m.group(2))
        seen = True
    return total if seen else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    fails = 0
    first = True
    with open(a.out, "w") as f:
        f.write("t," + ",".join(KEYS) + "\n")
        while not os.path.exists(a.stop_file):
            try:
                text = scrape(a.url)
                fails = 0
            except Exception:
                fails += 1
                if fails > 10:
                    break
                time.sleep(a.interval)
                continue
            if first:
                with open(a.out + ".first_metrics.txt", "w") as g:
                    g.write(text)
                first = False
            f.write(f"{time.time():.3f}," + ",".join(f"{value(text, k):g}" for k in KEYS.values()) + "\n")
            f.flush()
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
