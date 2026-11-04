"""W2 integration client: fixed requests against a running `vllm serve`.

Requests: real ChartQA / TextVQA images (local parquet), single-image and
two-image prompts, greedy decoding, sent one at a time (fixed batching), every
request twice (the second pass exercises prefix-cache hits).
For each request it records the output text, prompt_tokens and the expected
number of visual tokens per image computed with the HF image processor
(same `size` kwargs as the server) so the runner can check placeholder counts.

Usage: python integration_client.py --model NAME --model-dir DIR --out out.json
         [--data-root $VTS_ROOT/data/lmms-lab] [--longest-edge 1003520]
"""
import argparse
import base64
import glob
import io
import json
import os
import urllib.request

import pyarrow.parquet as pq
from PIL import Image


def load_images(root, n_chart=5, n_text=5):
    """[(image bytes, question)]: first n_chart ChartQA test and n_text TextVQA val rows
    (each source read from its first parquet shard)."""
    out = []
    for pattern, n in ((f"{root}/ChartQA/data/test-*.parquet", n_chart),
                       (f"{root}/textvqa/data/validation-*.parquet", n_text)):
        for r in pq.read_table(sorted(glob.glob(pattern))[0]).slice(0, n).to_pylist():
            out.append((r["image"]["bytes"], r["question"]))
    return out


def to_png_b64(raw):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), im


def chat(base_url, model, content, max_tokens, timeout=600):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0, "logprobs": True, "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    content_lp = (d["choices"][0].get("logprobs") or {}).get("content") or []
    lp = [(t["token"], t["logprob"]) for t in content_lp]
    top = [[(c["token"], c["logprob"]) for c in t.get("top_logprobs") or []] for t in content_lp]
    return d["choices"][0]["message"]["content"], d["usage"]["prompt_tokens"], lp, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8041")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=os.path.join(os.environ.get("VTS_ROOT", "."), "data/lmms-lab"))
    ap.add_argument("--longest-edge", type=int, default=1003520)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--n-chart", type=int, default=5)
    ap.add_argument("--n-text", type=int, default=5)
    ap.add_argument("--single-only", action="store_true")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    from transformers import AutoImageProcessor
    iproc = AutoImageProcessor.from_pretrained(args.model_dir)
    size = {"longest_edge": args.longest_edge, "shortest_edge": 65536}
    merge = iproc.merge_size

    imgs = load_images(args.data_root, args.n_chart, args.n_text)
    requests = [([raw], q + " Answer briefly.") for raw, q in imgs]
    if not args.single_only:  # two-image prompts: chart + scene
        for i in range(3):
            requests.append(([imgs[i][0], imgs[args.n_chart + i][0]], "Describe both images in one sentence each."))

    results = []
    for rep in range(args.passes):
        for ridx, (raws, text) in enumerate(requests):
            content, grids = [], []
            for raw in raws:
                b64, im = to_png_b64(raw)
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}})
                g = iproc(images=[im], size=size, return_tensors="pt")["image_grid_thw"][0].tolist()
                grids.append([int(g[1]) // merge, int(g[2]) // merge])
            content.append({"type": "text", "text": text})
            out, pt, lp, top = chat(args.base_url, args.model, content, args.max_tokens)
            results.append({"rep": rep, "req": ridx, "prompt_tokens": pt, "grids": grids, "output": out,
                            "logprobs": lp, "top_logprobs": top})
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
