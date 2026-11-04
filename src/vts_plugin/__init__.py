"""VLM-TokenServe: out-of-tree vLLM plugin for image visual-token compression.

Loaded through the `vllm.general_plugins` entry point in every vLLM process.
Does nothing unless VTS_METHOD is set (see config.py). Two diagnostics can be
turned on independently of the method, also for uncompressed runs:
VTS_VIT_TRACE=<dir> logs every vision-encoder call (vittrace.py) and
VTS_STEP_TRACE=<dir> logs every engine step (steptrace.py).

Supported models: Qwen3-VL (`Qwen3VLForConditionalGeneration`) and Qwen3.5
(`Qwen3_5ForConditionalGeneration`), image inputs only.
"""
import logging
import os

__version__ = "0.2.0"
REQUIRED_VLLM = "0.32.0"
# A child of vLLM's "vllm" logger: same format, level (VLLM_LOGGING_LEVEL) and handlers.
logger = logging.getLogger("vllm.vts_plugin")


def register() -> None:
    """vLLM plugin hook: install traces, then (if enabled) swap in the pruning model classes."""
    from . import steptrace, vittrace
    from .config import get_config

    if vittrace.maybe_install():
        logger.info("vts_plugin %s: ViT trace on -> %s", __version__, os.environ["VTS_VIT_TRACE"])
    if steptrace.maybe_install():
        logger.info("vts_plugin %s: step trace on -> %s", __version__, os.environ["VTS_STEP_TRACE"])
    cfg = get_config()
    if not cfg.enabled:
        return
    import vllm
    from vllm import ModelRegistry
    from vllm.config.multimodal import MultiModalConfig

    if vllm.__version__ != REQUIRED_VLLM:
        raise RuntimeError(f"vts_plugin needs vllm=={REQUIRED_VLLM}, found {vllm.__version__}")

    # Turn on vLLM's multimodal-pruning path (built for video EVS) for images: the model
    # then returns per-token positions and the V2 runner recomputes M-RoPE (models.py).
    if not getattr(MultiModalConfig, "_vts_patched", False):
        original = MultiModalConfig.is_multimodal_pruning_enabled

        def is_multimodal_pruning_enabled(self):
            return True if get_config().prunes else original(self)

        MultiModalConfig.is_multimodal_pruning_enabled = is_multimodal_pruning_enabled
        MultiModalConfig._vts_patched = True

    ModelRegistry.register_model(
        "Qwen3VLForConditionalGeneration",
        "vts_plugin.models:VTSQwen3VLForConditionalGeneration")
    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "vts_plugin.models:VTSQwen3_5ForConditionalGeneration")
    logger.info("vts_plugin %s registered: %s", __version__, cfg)
