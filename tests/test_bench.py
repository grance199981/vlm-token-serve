"""vts-bench (CPU only): the new CLI must reproduce the measurement protocol and statistics of the
experiment scripts that produced the published numbers. Old and new code run on the same inputs."""
import csv
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vts_bench import summarize as S  # noqa: E402
from vts_bench.accuracy import AccuracyRun, lmms_eval_cmd  # noqa: E402
from vts_bench.load import LoadRun, bench_cmd, point_name, server_spec  # noqa: E402
from vts_bench.presets import WORKLOADS, parse_method  # noqa: E402
from vts_bench.server import ServerSpec, build_env, build_serve_cmd  # noqa: E402


def load_script(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shell_function_body(rel, name):
    text = (REPO / rel).read_text(encoding="utf-8")
    m = re.search(rf"^{name}\(\) {{.*?\n}}", text, re.S | re.M)
    assert m, (rel, name)
    return m.group(0)


ROOT = Path("/r")


# --- protocol ----------------------------------------------------------------------------------

def test_seeds_and_prompt_counts_match_the_runners():
    """experiments/w4/run_w4_serving.sh: N = clamp(8C, MINN=32, 256), seed 400000 + rep*1e4 +
    budget_index*1e3 + C*10 + WID (P=1, D=2); run_k1.sh: N = max(4C, 256), seed 300000 + rep*1e4 + C."""
    P, D, K1 = WORKLOADS["P"], WORKLOADS["D"], WORKLOADS["K1"]
    for c in (1, 8, 32, 64, 128):
        assert P.num_prompts(c) == min(max(c * 8, 32), 256)
        assert K1.num_prompts(c) == max(c * 4, 256)
        for rep in (1, 2, 3):
            for bi in (0, 1):
                assert P.seed(rep, c, bi) == 400000 + rep * 10000 + bi * 1000 + c * 10 + 1
                assert D.seed(rep, c, bi) == 400000 + rep * 10000 + bi * 1000 + c * 10 + 2
            assert K1.seed(rep, c) == 300000 + rep * 10000 + c
    seeds = {P.seed(r, c, b) for r in (1, 2, 3) for c in (1, 8, 32, 64) for b in (0, 1)}
    assert len(seeds) == 24  # unique per point: no prefix-cache sharing across points


def test_bench_command_matches_w4_runner():
    """Every `vllm bench serve` flag of the W4 bench() function appears with the same value."""
    run = LoadRun(ROOT, Path("/o"), WORKLOADS["P"], ["M"], [parse_method("none", 1.0)], [64])
    cmd = bench_cmd(run, "M", 256, 64, 401641, result=Path("/o/results/x.json"))
    body = shell_function_body("experiments/w4/run_w4_serving.sh", "bench")
    for flag in ("--backend", "--endpoint", "--dataset-name", "--random-input-len", "--random-range-ratio",
                 "--random-mm-base-items-per-request", "--random-mm-num-mm-items-range-ratio",
                 "--temperature", "--request-rate"):
        old = re.search(re.escape(flag) + r" (\S+)", body).group(1)
        old = {"$ITEMS": "4", "$OUT": "32"}.get(old, old)  # the runner's values for WORKLOAD=P
        assert cmd[cmd.index(flag) + 1] == old, flag
    assert cmd[cmd.index("--random-output-len") + 1] == "32"  # OUT for P
    assert json.loads(cmd[cmd.index("--random-mm-limit-mm-per-prompt") + 1]) == {"image": 4, "video": 0}
    assert cmd[cmd.index("--random-mm-bucket-config") + 1] == "{(896, 896, 1): 1.0}"
    assert json.loads(cmd[cmd.index("--extra-body") + 1]) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "--ignore-eos" in cmd and "--save-result" in cmd


def test_server_flags_match_runners():
    none = parse_method("none", 0.5)
    p = build_serve_cmd(server_spec(LoadRun(ROOT, Path("/o"), WORKLOADS["P"], ["M"], [none], [1]),
                                    "M", none, "2048"))
    s = " ".join(p)
    for frag in ("--dtype bfloat16", "--gpu-memory-utilization 0.90", "--max-model-len 16384",
                 '--limit-mm-per-prompt {"image": 4, "video": 0}', "--max-num-batched-tokens 2048", "--seed 0"):
        assert frag in s, frag
    k1 = build_serve_cmd(server_spec(LoadRun(ROOT, Path("/o"), WORKLOADS["K1"], ["M"], [none], [1],
                                             kv_bytes=4294967296), "M", none, "default"))
    s = " ".join(k1)
    for frag in ("--kv-cache-memory-bytes 4294967296", "--max-model-len 4096", "--max-num-seqs 128",
                 '--limit-mm-per-prompt {"image": 2, "video": 0}'):
        assert frag in s, frag
    assert "--max-num-batched-tokens" not in s
    with pytest.raises(ValueError):
        server_spec(LoadRun(ROOT, Path("/o"), WORKLOADS["K1"], ["M"], [none], [1]), "M", none, "default")


def test_method_environment_is_isolated():
    base = {"PATH": "/bin", "VTS_METHOD": "rate", "VTS_KEEP": "0.25", "VTS_DIVPRUNE_IMPL": "triton"}
    env = build_env(ServerSpec(ROOT, "M", parse_method("none", 0.5)), base)
    assert not any(k.startswith("VTS_") for k in env) and env["PATH"] == "/bin"  # B0 really is plugin-off
    env = build_env(ServerSpec(ROOT, "M", parse_method("divprune_triton", 0.5)), base)
    assert (env["VTS_METHOD"], env["VTS_KEEP"], env["VTS_DIVPRUNE_IMPL"]) == ("divprune", "0.5", "triton")
    env = build_env(ServerSpec(ROOT, "M", parse_method("random", 0.25)), base)
    assert "VTS_DIVPRUNE_IMPL" not in env and env["VTS_KEEP"] == "0.25"


def test_point_names_parse_back():
    m = parse_method("downsample", 0.5)
    name = point_name("Qwen3-VL-4B-Instruct", WORKLOADS["P"], m, "2048", 64, 2) + ".json"
    assert name == "Qwen3-VL-4B-Instruct_P_downsample_k0.5_b2048_c64_rep2.json"
    old = load_script("experiments/w4/serve_summary.py")
    assert old.NAME.match(name).groupdict() == {k: v for k, v in S.POINT.match(name).groupdict().items()}
    assert S.POINT.match(point_name("M", WORKLOADS["K1"], parse_method("none", 1), "default", 128, 1) + ".json")


def test_lmms_eval_command_matches_w3_runner():
    run = AccuracyRun(ROOT, Path("/o"), ["M"], [parse_method("none", 1.0)], ["vts_textvqa_val"], port=8061)
    cmd = lmms_eval_cmd(run, "M", "vts_textvqa_val", Path("/o/x"))
    text = (REPO / "experiments/w3/run_w3.sh").read_text(encoding="utf-8")
    old_ma = re.search(r'MA="([^"]+)"', text).group(1).replace("$M", "M").replace("$PORT", "8061")
    assert cmd[cmd.index("--model_args") + 1] == old_ma
    assert cmd[cmd.index("--limit") + 1] == "1500"  # W3 TextVQA limit
    assert "--limit" not in lmms_eval_cmd(run, "M", "vts_chartqa", Path("/o/x"))


# --- statistics: identical values to the scripts behind the published tables ---------------------

def test_bootstrap_identical_to_w1_and_w3_scripts():
    rng = random.Random(7)
    xs = [float(rng.random() < 0.8) for _ in range(500)]
    diffs = [rng.choice([-1.0, 0.0, 0.0, 1.0]) for _ in range(400)]
    assert S.bootstrap_ci(xs) == load_script("experiments/w1/summarize_accuracy.py").bootstrap(xs)
    assert S.bootstrap_ci(diffs) == load_script("experiments/w3/summarize_w3.py").boot(diffs)


def test_steady_state_identical_to_k1_script(tmp_path):
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "running", "waiting", "kv_usage", "preemptions"])
        for i in range(50):
            w.writerow([i, 100 + i % 7, i % 3, 0.5 + 0.01 * (i % 5), i // 10])
    assert S.steady_state(str(p)) == load_script("experiments/w4/k1_steady.py").steady(str(p))


def _bench_json(path, req_s, tpot, ttft=100.0):
    json.dump({"failed": 0, "completed": 10, "num_prompts": 10, "request_throughput": req_s,
               "total_input_tokens": 1000, "duration": 5.0, "p50_ttft_ms": ttft, "p99_ttft_ms": 2 * ttft,
               "p50_tpot_ms": tpot}, open(path, "w"))


def test_gains_identical_to_w4_summary(tmp_path):
    rng = random.Random(3)
    for rep in (1, 2, 3):
        for method, keep, scale in (("none", "1.0", 1.0), ("downsample", "0.5", 2.0), ("rate", "0.5", 1.5)):
            for c in (8, 64):
                _bench_json(tmp_path / f"M_P_{method}_k{keep}_b2048_c{c}_rep{rep}.json",
                            scale * (1 + 0.05 * rng.random()), 20 / scale * (1 + 0.05 * rng.random()))
    old = load_script("experiments/w4/serve_summary.py")
    old_rows = {(r["method"], r["conc"]): r for r in old.summarize(old.load([str(tmp_path)]))}
    new_rows = S.serve_gains(S.load_serve_points([str(tmp_path)]))
    assert len(new_rows) == len(old_rows) == 6
    for r in new_rows:
        o = old_rows[(r["method"], r["conc"])]
        for k in ("req_s", "tpot_p50", "gain_req_s", "gain_req_s_range", "tpot_change_pct"):
            assert r.get(k) == o.get(k), k
    assert any(r.get("gain_req_s", 0) > 1.9 for r in new_rows)  # the fixture really has a gain


def test_paired_delta_identical_to_w3_summary(tmp_path):
    rng = random.Random(5)
    for name in ("b0", "cfg"):
        d = tmp_path / name
        d.mkdir()
        with open(d / "x_samples_vts_chartqa.jsonl", "w") as f:
            for i in range(300):
                score = float(rng.random() < (0.8 if name == "b0" else 0.7))
                f.write(json.dumps({"doc_id": i, "relaxed_overall": score}) + "\n")
    new = S.paired_delta(S.load_samples(str(tmp_path / "b0"), "vts_chartqa"),
                         S.load_samples(str(tmp_path / "cfg"), "vts_chartqa"))
    old = load_script("experiments/w3/summarize_w3.py")
    b0 = old.load_samples(str(tmp_path / "b0"), "vts_chartqa")
    cur = old.load_samples(str(tmp_path / "cfg"), "vts_chartqa")
    d = [cur[i][0] - b0[i][0] for i in sorted(b0)]
    lo, hi = old.boot(d)
    assert new["n"] == 300 and new["delta"] == 100 * sum(d) / len(d)
    assert (new["ci_lo"], new["ci_hi"]) == (100 * lo, 100 * hi)
    assert new["delta"] < 0  # the fixture really has a drop


def test_cli_rejects_unknown_preset_and_method_combinations():
    from vts_bench import cli
    with pytest.raises(SystemExit):
        cli.main(["serve", "--out", "/o", "--preset", "X"])
