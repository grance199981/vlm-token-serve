"""CPU analysis of the D1 (budget 2048) crash in vllm/multimodal/video_prune/evs.py:337.

Hypothesis H: with transformers>=5.10 vLLM keeps the prompt's outer
<|vision_start|> around the expanded Qwen3-VL video (qwen3_vl.py
`_expands_only_video_token`). recompute_mrope_positions() takes the first
vision_start at/after the cursor as frame 0's and subtracts the timestamp-token
count, so it starts writing num_ts + 1 tokens too early: a negative index (crash)
with a short text prefix, silently wrong positions with a longer one. Chunking is
not needed.

Control: the same video in the old layout (no outer markers) must give correct
positions. Also checks that the outer-marker layout reproduces the measured D2
prompt length (11671 tokens), i.e. that it is what vLLM actually built.

Usage: python evs_recompute_cpu.py --model <Qwen3-VL dir>
"""
import argparse
import json

import torch
from transformers import AutoConfig, AutoTokenizer

from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.video_prune import evs as _evs

compute_retained_tokens_count = _evs.compute_retained_tokens_count
recompute_mrope_positions = _evs.recompute_mrope_positions

FRAMES, FPS, TEMPORAL = 32, 2.0, 2
TOKENS_PER_FRAME = 720  # 46080 patches / 16 temporal frames / 2x2 merge (D2 log)
PRUNE_Q = 0.75
MEASURED_D2_PROMPT_LEN = 11671  # d2_noprune_b16384.json prompt_tokens


def timestamps():
    t = [i / FPS for i in range(FRAMES)]
    return [(t[i] + t[i + TEMPORAL - 1]) / 2 for i in range(0, FRAMES, TEMPORAL)]


def repl(tok, cfg, per_frame):
    return Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=per_frame, timestamps=timestamps(), tokenizer=tok,
        vision_start_token_id=cfg.vision_start_token_id, vision_end_token_id=cfg.vision_end_token_id,
        video_token_id=cfg.video_token_id).full


def model_side_positions(ids, cfg):
    """(5, L) channels as the model appends them: t,h,w for video tokens, a
    running index for text tokens, ch3 = is_vision_start, ch4 = is_video."""
    ids = torch.tensor(ids)
    pos = torch.zeros(5, len(ids), dtype=torch.long)
    vid = ids.eq(cfg.video_token_id)
    pos[3] = ids.eq(cfg.vision_start_token_id).long()
    pos[4] = vid.long()
    # distinct, non-identity values per channel: an identity ramp would hide a shifted write
    i = torch.arange(len(ids))
    pos[0], pos[1], pos[2] = 2 * i, 3 * i + 1, 5 * i + 2
    return pos


def run(input_ids, mm_pos, num_computed, mm_offset):
    n = len(input_ids)
    base_pos = torch.arange(n).repeat(3, 1)
    try:
        out, _ = recompute_mrope_positions(torch.tensor(input_ids), [mm_pos], base_pos.clone(),
                                           num_computed, CFG.vision_start_token_id,
                                           CFG.image_token_id, CFG.video_token_id)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    # Expected: the placeholder's first column goes to input index mm_offset + num_computed_in_item.
    start = mm_offset + max(0, num_computed - mm_offset)
    want = mm_pos[0:3] + base_pos[-1, start - 1] + 1
    got = out[:, start:start + mm_pos.shape[1]]
    ok = got.shape == want.shape and bool(torch.equal(got, want))
    # where did the write actually land? search for the first column of want
    shift = None
    if not ok:
        hits = ((out[0] == want[0, 0]) & (out[1] == want[1, 0]) & (out[2] == want[2, 0])).nonzero()
        shift = int(hits[0]) - start if len(hits) else None
    prefix_untouched = bool(torch.equal(out[:, :start], base_pos[:, :start]))
    n_prefix_overwritten = int((out[:, :start] != base_pos[:, :start]).any(0).sum())
    return {"error": None, "positions_correct": ok, "write_shift": shift, "prefix_untouched": prefix_untouched,
            "n_text_tokens_before_media_overwritten": n_prefix_overwritten}


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--evs-file", help="load recompute_mrope_positions from this evs.py (e.g. a PR head)")
    a = ap.parse_args()
    global recompute_mrope_positions
    if a.evs_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location("evs_alt", a.evs_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        recompute_mrope_positions = mod.recompute_mrope_positions
    tok = AutoTokenizer.from_pretrained(a.model)
    CFG = AutoConfig.from_pretrained(a.model)
    vs, ve = CFG.vision_start_token_id, CFG.vision_end_token_id
    prefix = tok.encode("<|im_start|>user\n", add_special_tokens=False)
    long_prefix = tok.encode("<|im_start|>system\nYou are a helpful assistant. Answer briefly and "
                             "precisely.<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
    suffix = tok.encode("Describe this video in one sentence.<|im_end|>\n<|im_start|>assistant\n",
                        add_special_tokens=False)
    report = {"evs_source": a.evs_file or _evs.__file__}

    # 1) which layout did vLLM build? compare with the measured D2 prompt length
    full = repl(tok, CFG, [TOKENS_PER_FRAME] * len(timestamps()))
    report["d2_prompt_len_measured"] = MEASURED_D2_PROMPT_LEN
    report["d2_len_outer_markers"] = len(prefix) + 1 + len(full) + 1 + len(suffix)
    report["d2_len_no_outer_markers"] = len(prefix) + len(full) + len(suffix)
    report["timestamp_tokens_frame0"] = len(tok.encode(f"<{timestamps()[0]:.1f} seconds>", add_special_tokens=False))

    # 2) EVS layouts: prompt keeps all retained tokens in frame 0; model side spreads them
    n_frames = len(timestamps())
    kept = compute_retained_tokens_count(tokens_per_frame=TOKENS_PER_FRAME, num_frames=n_frames, q=PRUNE_Q)
    prompt_repl = repl(tok, CFG, [kept] + [0] * (n_frames - 1))
    rest = kept - TOKENS_PER_FRAME
    spread = [TOKENS_PER_FRAME] + [rest // (n_frames - 1) + (i < rest % (n_frames - 1)) for i in range(n_frames - 1)]
    model_repl = repl(tok, CFG, spread)
    assert len(prompt_repl) == len(model_repl) and sum(spread) == kept
    mm_pos = model_side_positions(model_repl, CFG)
    report["evs_kept_tokens"] = kept
    report["placeholder_len"] = len(prompt_repl)

    cases = {}
    for name, pre, outer in [("outer_markers_short_prefix", prefix, True),
                             ("outer_markers_long_prefix", long_prefix, True),
                             ("CONTROL_no_outer_markers_short_prefix", prefix, False),
                             ("CONTROL_no_outer_markers_long_prefix", long_prefix, False)]:
        ids = pre + ([vs] if outer else []) + prompt_repl + ([ve] if outer else []) + suffix
        off = len(pre) + (1 if outer else 0)
        # whole prompt in one step (no chunking) and the first 2048-token chunk
        cases[name + "/one_step"] = run(ids, mm_pos, 0, off)
        cases[name + "/chunk1_2048"] = dict(run(ids, mm_pos[:, :2048 - off], 0, off), mm_cols=2048 - off)
    # negative control for the checker: a correct write shifted by one token must be flagged
    ids = prefix + prompt_repl + suffix
    off = len(prefix)
    base_pos = torch.arange(len(ids)).repeat(3, 1)
    shifted = base_pos.clone()
    shifted[:, off - 1:off - 1 + mm_pos.shape[1]] = mm_pos[0:3] + base_pos[-1, off - 1] + 1
    want = mm_pos[0:3] + base_pos[-1, off - 1] + 1
    cases["CHECKER_negative_control_shift_by_1_detected"] = not torch.equal(shifted[:, off:off + mm_pos.shape[1]], want)
    report["cases"] = cases
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
