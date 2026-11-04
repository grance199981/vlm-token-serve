"""HF transformers reference for the W0 digit-reading smoke test (no vLLM).

Same synthetic images and prompt as smoke_client.py, greedy decoding. Used to
tell a vLLM-pipeline problem from a model limitation.
"""
import argparse
import base64
import io
import json
import random

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from smoke_client import make_image

QUESTION = "What number is written in the image? Answer with the number only."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    rows = []
    for seed in args.seeds:
        number = random.Random(seed).randint(1000, 9999)
        img = Image.open(io.BytesIO(base64.b64decode(make_image(number)))).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": QUESTION}]}]
        inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt", enable_thinking=False).to(model.device)
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        rows.append({"seed": seed, "number": number, "answer": text, "ok": str(number) in text,
                     "prompt_tokens": int(inputs["input_ids"].shape[1])})
    print(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
