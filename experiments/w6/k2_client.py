"""K2 multi-turn image-chat load client (docs/plans/K2_PLAN.md §2).

S sessions take turns round-robin: every session's turn t is sent (at most --conc in flight) before
any session's turn t+1, so between two turns of one session there are the other S-1 sessions'
requests. Turn 1 = one synthetic image (unique per session) + a text line; later turns = one text
line; the full history, including the image and the model's own replies, is re-sent every turn.
One JSON line per request: session, turn, prompt / cached / completion tokens (served with
--enable-prompt-tokens-details), TTFT and end-to-end time.

Usage: python k2_client.py --url http://127.0.0.1:8096 --model <served name> --sessions 32 \
         --seed 500032 --out <jsonl> [--turns 4] [--conc 8] [--out-tokens 64] [--side 896]
"""
import argparse
import asyncio
import base64
import io
import json
import time

import numpy as np

TURN1 = "Session {sid}. Describe this image in detail."
FOLLOW = "Question {turn}: tell me more about region {turn} of the image."


def image_data_url(seed, side):
    rng = np.random.default_rng(seed)
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (side, side, 3), dtype=np.uint8)).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def user_message(turn, sid, url):
    if turn == 1:
        return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                            {"type": "text", "text": TURN1.format(sid=sid)}]}
    return {"role": "user", "content": FOLLOW.format(turn=turn)}


def parse_sse_line(line):
    """Returns ("delta", text) / ("usage", dict) / ("done", None) / (None, None)."""
    line = line.strip()
    if not line.startswith("data:"):
        return None, None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "done", None
    obj = json.loads(payload)
    if obj.get("usage"):
        return "usage", obj["usage"]
    for ch in obj.get("choices") or []:
        text = (ch.get("delta") or {}).get("content")
        if text:
            return "delta", text
    return None, None


async def one_request(session, url, model, messages, out_tokens):
    body = {"model": model, "messages": messages, "max_tokens": out_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    ttft, text, usage = None, [], {}
    async with session.post(url + "/v1/chat/completions", json=body) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            kind, val = parse_sse_line(raw.decode())
            if kind == "delta":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(val)
            elif kind == "usage":
                usage = val
    return {"ttft_ms": 1000 * ttft if ttft is not None else None, "e2e_ms": 1000 * (time.perf_counter() - t0),
            "text": "".join(text), "usage": usage}


async def run(a):
    import aiohttp

    urls = [image_data_url(a.seed * 1000 + s, a.side) for s in range(a.sessions)]
    hist = [[] for _ in range(a.sessions)]
    sem = asyncio.Semaphore(a.conc)
    timeout = aiohttp.ClientTimeout(total=600)
    out = open(a.out, "w")
    async with aiohttp.ClientSession(timeout=timeout) as http:
        for turn in range(1, a.turns + 1):
            async def go(s):
                msgs = hist[s] + [user_message(turn, f"{a.seed}-{s}", urls[s])]
                async with sem:
                    t_start = time.time()
                    try:
                        r = await one_request(http, a.url, a.model, msgs, a.out_tokens)
                        err = None
                    except Exception as e:  # noqa: BLE001 - recorded, the run goes on
                        r, err = {"ttft_ms": None, "e2e_ms": None, "text": "", "usage": {}}, repr(e)
                hist[s] = msgs + [{"role": "assistant", "content": r["text"]}]
                u = r["usage"] or {}
                details = u.get("prompt_tokens_details") or {}
                out.write(json.dumps({"seed": a.seed, "sessions": a.sessions, "session": s, "turn": turn,
                                      "t_start": round(t_start, 4), "prompt_tokens": u.get("prompt_tokens"),
                                      "cached_tokens": details.get("cached_tokens"),
                                      "completion_tokens": u.get("completion_tokens"),
                                      "ttft_ms": r["ttft_ms"], "e2e_ms": r["e2e_ms"], "error": err}) + "\n")
            await asyncio.gather(*(go(s) for s in range(a.sessions)))
            out.flush()
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument("--side", type=int, default=896)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
