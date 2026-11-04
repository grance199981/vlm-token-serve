"""W0 smoke test against a running `vllm serve` (OpenAI-compatible API).

Renders a synthetic image containing a random 4-digit number, asks the model to
read it, and checks that (a) the image actually reached the model (prompt
tokens with the image exceed the text-only prompt by a margin) and (b) the
answer contains the number. A text-only request with the same question is the
negative control: it must not contain the number.
"""
import argparse
import base64
import io
import json
import random
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont


def make_image(number, size=(448, 224)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 120)
    except OSError:
        font = ImageFont.load_default(size=120)
    d.text((30, 40), str(number), fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def chat(base_url, model, content, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    out["_latency_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    number = random.Random(args.seed).randint(1000, 9999)
    question = "What number is written in the image? Answer with the number only."
    with_img = chat(args.base_url, args.model, [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + make_image(number)}},
        {"type": "text", "text": question}], args.max_tokens)
    text_only = chat(args.base_url, args.model, question, args.max_tokens)

    ans = with_img["choices"][0]["message"]["content"] or ""
    ctrl = text_only["choices"][0]["message"]["content"] or ""
    pt_img = with_img["usage"]["prompt_tokens"]
    pt_txt = text_only["usage"]["prompt_tokens"]
    checks = {
        "image_tokens_reached_model": pt_img - pt_txt >= 32,
        "answer_contains_number": str(number) in ans,
        "negative_control_lacks_number": str(number) not in ctrl,
    }
    report = {"number": number, "answer": ans, "control_answer": ctrl,
              "prompt_tokens_with_image": pt_img, "prompt_tokens_text_only": pt_txt,
              "latency_s_with_image": round(with_img["_latency_s"], 3), "checks": checks}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
