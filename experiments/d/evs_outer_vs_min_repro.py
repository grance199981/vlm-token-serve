"""Minimal CPU repro for vllm-project/vllm#62937.

No model, no GPU: only torch + vllm. Lays out a Qwen3-VL EVS video placeholder the
way vLLM builds it under transformers>=5.10 (outer <|vision_start|>/<|vision_end|>
kept around the per-frame "timestamp VS video VE" blocks) and calls
vllm.multimodal.video_prune.evs.recompute_mrope_positions on the first prefill.
"""
import torch

from vllm.multimodal.video_prune.evs import recompute_mrope_positions

TEXT, TS, VS, VE, VIDEO, IMAGE = 1, 2, 777, 778, 888, 999
NUM_TS = 6  # e.g. "<0.2 seconds>" is 6 tokens for the Qwen3-VL tokenizer
FRAMES, KEPT = 4, 10  # EVS keeps 10 video tokens in total


def frame_block(n_video):
    return [TS] * NUM_TS + [VS] + [VIDEO] * n_video + [VE]


# prompt side: EVS puts all kept tokens on frame 0 (real per-frame split is unknown yet)
placeholder = frame_block(KEPT) + sum((frame_block(0) for _ in range(FRAMES - 1)), [])
# model side: the same length, but the kept tokens spread over the frames
model_side = sum((frame_block(n) for n in (4, 2, 2, 2)), [])
assert len(model_side) == len(placeholder)

ids = torch.tensor(model_side)
mm_pos = torch.zeros(5, len(model_side), dtype=torch.long)
i = torch.arange(len(model_side))
mm_pos[0], mm_pos[1], mm_pos[2] = 2 * i, 3 * i + 1, 5 * i + 2  # any distinct values
mm_pos[3] = ids.eq(VS).long()     # is_vision_start
mm_pos[4] = ids.eq(VIDEO).long()  # is_video


def run(prefix_len, outer_markers):
    prompt = [TEXT] * prefix_len + ([VS] if outer_markers else []) + placeholder \
        + ([VE] if outer_markers else []) + [TEXT] * 3
    start = prefix_len + int(outer_markers)  # where the placeholder really begins
    pos = torch.arange(len(prompt)).repeat(3, 1)
    try:
        out, _ = recompute_mrope_positions(torch.tensor(prompt), [mm_pos], pos.clone(), 0,
                                           VS, IMAGE, VIDEO)
    except RuntimeError as e:
        return f"RuntimeError: {str(e)[:110]}"
    want = mm_pos[0:3] + pos[-1, start - 1] + 1
    ok = torch.equal(out[:, start:start + mm_pos.shape[1]], want)
    touched = int((out[:, :start] != pos[:, :start]).any(0).sum())
    return f"positions correct={ok}, text tokens before the video overwritten={touched}"


for prefix_len, outer in [(3, True), (20, True), (3, False), (20, False)]:
    print(f"prefix={prefix_len:2d} outer_markers={outer!s:5}: {run(prefix_len, outer)}")
