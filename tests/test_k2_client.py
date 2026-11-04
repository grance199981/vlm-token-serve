"""K2 client (experiments/w6/k2_client.py) against a fake OpenAI streaming server: round-robin order,
history re-sent with the image and the model's replies, usage / cached tokens parsed, TTFT taken.
CPU only, no vLLM."""
import asyncio
import importlib.util
import json
import pathlib
import socket
import sys
import threading
import types

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("k2_client", ROOT / "experiments" / "w6" / "k2_client.py")
k2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k2)


def test_parse_sse_line():
    assert k2.parse_sse_line("data: [DONE]") == ("done", None)
    assert k2.parse_sse_line(": ping") == (None, None)
    d = {"choices": [{"delta": {"content": "hi"}}]}
    assert k2.parse_sse_line("data: " + json.dumps(d)) == ("delta", "hi")
    u = {"choices": [], "usage": {"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3}}}
    assert k2.parse_sse_line("data: " + json.dumps(u))[0] == "usage"
    assert k2.parse_sse_line("data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]})) == (None, None)


def test_images_unique_and_deterministic():
    a, b = k2.image_data_url(1, 64), k2.image_data_url(2, 64)
    assert a != b and a == k2.image_data_url(1, 64) and a.startswith("data:image/jpeg;base64,")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run_fake(port, log, ready):
    async def handler(req):
        body = await req.json()
        log.append(body)
        n = len(body["messages"])
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        await resp.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        for w in ("reply", f"-{n}"):
            await resp.write(("data: " + json.dumps({"choices": [{"delta": {"content": w}}]}) + "\n\n").encode())
        usage = {"prompt_tokens": 100 * n, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 10 * n}}
        await resp.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    loop = asyncio.new_event_loop()
    app = web.Application(client_max_size=64 * 2 ** 20)
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
    ready.set()
    loop.run_forever()


def test_round_robin_history_and_records(tmp_path):
    port, log, ready = free_port(), [], threading.Event()
    threading.Thread(target=run_fake, args=(port, log, ready), daemon=True).start()
    assert ready.wait(10)
    out = tmp_path / "k2.jsonl"
    a = types.SimpleNamespace(url=f"http://127.0.0.1:{port}", model="m", sessions=3, seed=7, out=str(out),
                              turns=3, conc=2, out_tokens=4, side=64)
    asyncio.run(k2.run(a))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 9 and len(log) == 9
    assert all(r["error"] is None for r in rows)
    # all of turn t before any of turn t+1 (requests as the server saw them)
    turns_seen = [(len(b["messages"]) + 1) // 2 for b in log]
    assert turns_seen == sorted(turns_seen) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    # turn 3 re-sends turn 1 (image first) and both earlier real replies
    last = [b for b in log if len(b["messages"]) == 5]
    for b in last:
        m = b["messages"]
        assert m[0]["content"][0]["type"] == "image_url" and m[1] == {"role": "assistant", "content": "reply-1"}
        assert m[3] == {"role": "assistant", "content": "reply-3"} and m[4]["role"] == "user"
        assert b["stream_options"] == {"include_usage": True} and b["ignore_eos"] is True
    # images differ across sessions (negative control: same image would collide in the prefix cache)
    assert len({b["messages"][0]["content"][0]["image_url"]["url"] for b in log if len(b["messages"]) == 1}) == 3
    r3 = [r for r in rows if r["turn"] == 3]
    assert {r["session"] for r in r3} == {0, 1, 2}
    assert all(r["prompt_tokens"] == 500 and r["cached_tokens"] == 50 and r["ttft_ms"] > 0 for r in r3)


def test_error_is_recorded(tmp_path):
    out = tmp_path / "k2.jsonl"
    a = types.SimpleNamespace(url=f"http://127.0.0.1:{free_port()}", model="m", sessions=2, seed=1, out=str(out),
                              turns=1, conc=2, out_tokens=4, side=32)
    asyncio.run(k2.run(a))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 2 and all(r["error"] and r["cached_tokens"] is None for r in rows)


def test_s_star():
    spec2 = importlib.util.spec_from_file_location("k2_summary", ROOT / "experiments" / "w6" / "k2_summary.py")
    sm = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(sm)
    assert sm.s_star([(8, 0.8), (16, 0.8), (32, 0.2)]) == pytest.approx(16 + 0.4 / 0.6 * 16)
    assert sm.s_star([(8, 0.8), (16, 0.4)]) == 16
    assert sm.s_star([(8, 0.8), (16, 0.7)]) == float("inf")  # never halves inside the grid


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
