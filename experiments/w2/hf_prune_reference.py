"""T5: HF-transformers reference for the plugin's pruning (no vLLM involved).

For the same requests as experiments/w2/integration_client.py (pass 0 only), it
runs HF Qwen3-VL / Qwen3.5 with:
  - image features from HF's own vision tower (incl. DeepStack levels),
  - the same token selection as the plugin, restricted to content-independent methods
    (keep_all, random, pool2x2) so tiny ViT numeric differences cannot change WHICH
    tokens are kept,
  - k placeholders per image and explicit M-RoPE position_ids (retained tokens at
    their original grid positions, "original" tail),
  - a manual greedy loop (48 tokens) recording chosen-token logprobs and top-5.
Output JSON has the same schema as integration_client.py, so the vLLM outputs can
be compared with experiments/w2/compare_hf_vllm.py.

Usage: python hf_prune_reference.py --model-dir DIR --method {keep_all,random,pool2x2}
         [--keep 0.5] --out out.json [--data-root $VTS_ROOT/data/lmms-lab]
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from integration_client import load_images, to_png_b64  # noqa: E402
from vts_plugin.config import VTSConfig, keep_count  # noqa: E402
from vts_plugin.mrope import initial_positions, recompute_image_positions  # noqa: E402
from vts_plugin.select import select  # noqa: E402


def build_requests(root, n_chart=5, n_text=5, single_only=False):
    imgs = load_images(root, n_chart, n_text)
    reqs = [([raw], q + " Answer briefly.") for raw, q in imgs]
    if not single_only:
        for i in range(3):
            reqs.append(([imgs[i][0], imgs[n_chart + i][0]], "Describe both images in one sentence each."))
    return reqs


@torch.no_grad()
def run_one(model, proc, cfg, raws, text, size, max_tokens, device, pos_mode="original"):
    pils = [to_png_b64(r)[1] for r in raws]
    messages = [{"role": "user", "content": [*({"type": "image"} for _ in pils), {"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    try:
        inputs = proc(text=[prompt], images=pils, images_kwargs={"size": size}, return_tensors="pt")
    except TypeError:
        inputs = proc(text=[prompt], images=pils, size=size, return_tensors="pt")
    inputs = inputs.to(device)
    core = model.model
    vis = core.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
    merge = core.visual.spatial_merge_size
    image_token_id = model.config.image_token_id
    ids = inputs["input_ids"][0].tolist()

    ds_levels = getattr(vis, "deepstack_features", None) or []
    new_pool, new_ds, kept_pos, spans = [], [[] for _ in ds_levels], [], []
    for i, grid in enumerate(inputs["image_grid_thw"].tolist()):
        _, h, w = grid
        gh, gw = h // merge, w // merge
        n = gh * gw
        feats = vis.pooler_output[i]
        levels = [lvl[i] for lvl in ds_levels]
        r = torch.arange(n, device=device)
        pos = torch.stack([torch.zeros_like(r), r // gw, r % gw, torch.full_like(r, gw), torch.full_like(r, gh)], 1)
        k = keep_count(cfg, n, gh, gw)
        widths = [feats.shape[1]] + [lv.shape[1] for lv in levels]
        cat = torch.cat([feats, *levels], dim=1)  # select all levels jointly, as the plugin does
        method = cfg.method if cfg.method != "none" else "keep_all"
        sel_feats, sel_pos = select(method, cat, pos, k=k, grid_h=gh, grid_w=gw, seed=cfg.seed)
        parts = torch.split(sel_feats, widths, dim=1)
        new_pool.append(parts[0])
        for j, p in enumerate(parts[1:]):
            new_ds[j].append(p)
        kept_pos.append(sel_pos)
        spans.append((n, k))

    # rebuild input ids: each run of n image tokens -> k tokens
    new_ids, it, img = [], 0, 0
    while it < len(ids):
        if ids[it] == image_token_id:
            n, k = spans[img]
            new_ids += [image_token_id] * k
            it += n
            img += 1
        else:
            new_ids.append(ids[it])
            it += 1
    ids_t = torch.tensor(new_ids, device=device)
    offsets = (ids_t == image_token_id).nonzero(as_tuple=True)[0].tolist()
    images, o = [], 0
    for (n, k), grid in zip(spans, inputs["image_grid_thw"].tolist()):
        images.append((offsets[o], k, grid[1] // merge, grid[2] // merge))
        o += k
    pos0, _ = initial_positions(len(new_ids), images)
    vs_id = model.config.vision_start_token_id
    chunk = [p.T.long() for p in kept_pos]
    if pos_mode == "placeholder":  # diagnostic: retained tokens keep the truncated-grid placeholders
        positions = pos0.to(device)
    else:
        positions, _delta = recompute_image_positions(ids_t, chunk, pos0.to(device), 0, vs_id, image_token_id,
                                                      "original")

    class Enc:  # minimal stand-in for BaseModelOutputWithDeepstackFeatures
        pass

    enc = Enc()
    enc.pooler_output = tuple(new_pool)
    enc.deepstack_features = [tuple(x) for x in new_ds]
    mm_type = (ids_t == image_token_id).int().unsqueeze(0)
    out = model(input_ids=ids_t.unsqueeze(0), position_ids=positions.unsqueeze(1),
                mm_encoder_outputs={"image": enc}, image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=mm_type, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1].float()
    next_pos = int(positions.max()) + 1
    toks, lps, tops = [], [], []
    tok = proc.tokenizer
    stop = (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
    for step in range(max_tokens):
        lp = torch.log_softmax(logits, -1)
        nxt = int(torch.argmax(lp))
        top = torch.topk(lp, 5)
        toks.append(nxt)
        lps.append((tok.decode([nxt]), float(lp[nxt])))
        tops.append([(tok.decode([int(t)]), float(v)) for v, t in zip(top.values, top.indices)])
        if nxt in stop:
            break
        p = torch.full((3, 1, 1), next_pos + step, device=device, dtype=torch.long)
        out = model(input_ids=torch.tensor([[nxt]], device=device), position_ids=p,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
    text_out = tok.decode([t for t in toks if t not in stop])
    return text_out, len(new_ids), lps, tops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--method", choices=["keep_all", "random", "pool2x2"], required=True)
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=os.path.join(os.environ.get("VTS_ROOT", "."), "data/lmms-lab"))
    ap.add_argument("--longest-edge", type=int, default=1003520)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--n-chart", type=int, default=5)
    ap.add_argument("--n-text", type=int, default=5)
    ap.add_argument("--single-only", action="store_true")
    ap.add_argument("--pos-mode", choices=["original", "placeholder"], default="original")
    args = ap.parse_args()
    device = "cuda:0"
    cfg = VTSConfig(method=args.method, keep=args.keep, seed=0)
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16,
                                                        attn_implementation="sdpa").to(device).eval()
    size = {"longest_edge": args.longest_edge, "shortest_edge": 65536}
    results = []
    for ridx, (raws, text) in enumerate(build_requests(args.data_root, args.n_chart, args.n_text,
                                                       args.single_only)):
        out, n_tok, lps, tops = run_one(model, proc, cfg, raws, text, size, args.max_tokens, device,
                                        args.pos_mode)
        results.append({"rep": 0, "req": ridx, "prompt_tokens": n_tok, "output": out,
                        "logprobs": lps, "top_logprobs": tops})
        print(ridx, n_tok, repr(out[:80]), flush=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
