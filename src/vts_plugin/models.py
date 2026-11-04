"""Qwen3-VL / Qwen3.5 subclasses that prune image tokens (imported lazily by vLLM).

Reuses vLLM 0.32's multimodal-pruning path (built for video EVS):
  1. the processor shrinks each image placeholder to `keep_count` tokens
     (`VTSQwen3VLProcessor._get_prompt_updates`), so the scheduler and the KV
     cache only ever see the kept tokens;
  2. the model runs the vision tower in profile-sized groups (`vitbatch.py`) and,
     in `_postprocess_image_embeds_evs`, appends 5 position channels
     [t, h, w, llm_grid_w, llm_grid_h] to every visual token, then selects k rows;
  3. the V2 model runner strips the channels and calls `recompute_mrope_positions`,
     which places the kept tokens at their original (t, h, w) M-RoPE positions.
`downsample` instead resizes images before the HF processor (no pruning).
See docs/W2_PLUGIN_DESIGN.md and docs/CODE_GUIDE_zh.md.
"""
import logging

import torch
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import PromptReplacement

from .config import get_config, keep_count
from .mrope import initial_positions, recompute_image_positions
from .resize import downscale
from .select import DIVPRUNE_TRITON_STATS, select
from .vitbatch import run_grouped, vit_patch_cap

logger = logging.getLogger("vllm.vts_plugin")
_POS_CH = 5  # position channels appended to each visual token by the EVS path


class VTSQwen3VLProcessor(Qwen3VLMultiModalProcessor):
    """Runs in the API-server process: decides how many placeholder tokens an image gets."""

    def _get_hf_mm_inputs(self, mm_items, hf_kwargs):
        result = super()._get_hf_mm_inputs(mm_items, hf_kwargs)
        cfg = get_config()
        if cfg.method == "downsample" and cfg.keep < 1.0:
            hf_data = result[0]
            if hf_data.get("images") is not None:
                hf_data["images"] = [downscale(im, cfg.keep) for im in hf_data["images"]]
        return result

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        cfg = get_config()
        if not cfg.prunes:
            return updates
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        merge = image_processor.merge_size

        def image_replacement(item_idx: int):
            grid = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            t, h, w = (int(x) for x in grid.tolist())
            gh, gw = h // merge, w // merge
            return [hf_processor.image_token_id] * keep_count(cfg, t * gh * gw, gh, gw)

        out = []
        for u in updates:
            if isinstance(u, PromptReplacement) and u.modality == "image":
                u = PromptReplacement(modality="image", target=u.target, replacement=image_replacement)
            out.append(u)
        return out


class VTSMixin:
    """Overrides shared by both architectures (put first in the MRO). Runs in EngineCore."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        mc = vllm_config.model_config
        self._vts_vit_cap = vit_patch_cap(mc.model, mc.get_multimodal_config().mm_processor_kwargs)
        self._vts_vit_splits = 0
        self._vts_impl_logged = False
        self._vts_fb_logged = False
        logger.info("vts_plugin: ViT patch cap per call = %d", self._vts_vit_cap)

    def _process_image_input(self, image_input):
        # Pruning makes the scheduler admit more patches per step than profiled
        # (see vitbatch.py); run the tower in profiled-size groups instead.
        if (not get_config().prunes or image_input["type"] == "image_embeds"
                or self.use_data_parallel):
            return super()._process_image_input(image_input)
        grid_thw = image_input["image_grid_thw"]
        embeds, n_calls = run_grouped(self.visual, image_input["pixel_values"], grid_thw,
                                      self._vts_vit_cap, self.visual.spatial_merge_size)
        if n_calls > 1:
            self._vts_vit_splits += 1
            if self._vts_vit_splits == 1 or self._vts_vit_splits % 100 == 0:
                logger.warning("vts_plugin: ViT batch of %d images / %d patches split into %d calls "
                               "(%d splits so far)", grid_thw.shape[0], int(grid_thw.prod(-1).sum()),
                               n_calls, self._vts_vit_splits)
        return embeds

    def _postprocess_image_embeds_evs(self, image_embeds_split, image_input):
        out = super()._postprocess_image_embeds_evs(image_embeds_split, image_input)
        cfg = get_config()
        if not cfg.prunes:
            return out
        merge = self.visual.spatial_merge_size
        grids = image_input["image_grid_thw"].tolist()
        score_dim = self.config.text_config.hidden_size  # main merger output (LLM input width)
        result = []
        for emb, (_t, h, w) in zip(out, grids):
            gh, gw = h // merge, w // merge
            feats, pos = emb[:, :-_POS_CH], emb[:, -_POS_CH:].clone()
            pos[:, 4] = gh  # spare channel: llm_grid_h (read only by our recompute)
            k = keep_count(cfg, feats.shape[0], gh, gw)
            new_feats, new_pos = select(cfg.method, feats, pos, k=k, grid_h=gh, grid_w=gw,
                                        seed=cfg.seed, score_dim=score_dim,
                                        divprune_impl=cfg.divprune_impl)
            if cfg.debug_pos_transpose:  # TEST ONLY negative control
                new_pos = new_pos.clone()
                new_pos[:, [1, 2]] = new_pos[:, [2, 1]]
            if new_feats.shape[0] != k:
                raise RuntimeError(f"vts: selected {new_feats.shape[0]} tokens, placeholder has {k}")
            result.append(torch.cat([new_feats, new_pos.to(new_feats.dtype)], dim=1))
        if cfg.method == "divprune" and cfg.divprune_impl == "triton":
            self._log_divprune_impl()
        return tuple(result)

    def _log_divprune_impl(self):
        """Log the first Triton call and the first fallback (the startup profile run uses
        the largest image, N > MAX_TOKENS, so it falls back)."""
        st = DIVPRUNE_TRITON_STATS
        new_tri = st["triton"] > 0 and not self._vts_impl_logged
        new_fb = st["fallback"] > 0 and not self._vts_fb_logged
        if new_tri or new_fb:
            self._vts_impl_logged |= new_tri
            self._vts_fb_logged |= new_fb
            logger.info("vts_plugin: divprune triton calls %d, torch fallbacks %d",
                        st["triton"], st["fallback"])

    @staticmethod
    def _get_mrope_input_positions(input_tokens, mm_features, config):
        """Initial M-RoPE positions of a prompt whose image placeholders are already pruned."""
        cfg = get_config()
        if not cfg.prunes or any(f.modality != "image" for f in mm_features):
            return Qwen3VLForConditionalGeneration._get_mrope_input_positions(
                input_tokens, mm_features, config)
        merge = config.vision_config.spatial_merge_size
        images = []
        for f in sorted(mm_features, key=lambda f: f.mm_position.offset):
            _t, h, w = (int(x) for x in f.data["image_grid_thw"].data.tolist())
            images.append((f.mm_position.offset, f.mm_position.length, h // merge, w // merge))
        return initial_positions(len(input_tokens), images, cfg.debug_pos_shift)

    def recompute_mrope_positions(self, input_ids, multimodal_embeddings, mrope_positions,
                                  num_computed_tokens):
        """Called by the V2 runner after the encoder: place kept tokens at their real positions."""
        cfg = get_config()
        device = multimodal_embeddings[0].device if len(multimodal_embeddings) else mrope_positions.device
        ids = (input_ids.to(device=device, dtype=torch.long) if isinstance(input_ids, torch.Tensor)
               else torch.tensor(input_ids, device=device, dtype=torch.long))
        if not cfg.prunes or bool(ids.eq(self.config.video_token_id).any()):
            return super().recompute_mrope_positions(
                input_ids, multimodal_embeddings, mrope_positions, num_computed_tokens)
        cleaned = [mm[:, :-_POS_CH] for mm in multimodal_embeddings]
        chunk_pos = [mm[:, -_POS_CH:].permute(1, 0).long() for mm in multimodal_embeddings]
        positions, delta = recompute_image_positions(
            ids, chunk_pos, mrope_positions.to(device), num_computed_tokens,
            self.config.vision_start_token_id, self.config.image_token_id, cfg.tail)
        return cleaned, positions, delta


@MULTIMODAL_REGISTRY.register_processor(
    VTSQwen3VLProcessor, info=Qwen3VLProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder)
class VTSQwen3VLForConditionalGeneration(VTSMixin, Qwen3VLForConditionalGeneration):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    VTSQwen3VLProcessor, info=Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder)
class VTSQwen3_5ForConditionalGeneration(VTSMixin, Qwen3_5ForConditionalGeneration):
    pass


logger.info("vts_plugin: model classes loaded, config=%s", get_config())
