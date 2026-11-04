"""Downsampling baseline: downscale images before the HF processor (no vLLM imports).

`downscale(image, keep)` shrinks both sides by sqrt(keep), so the pixel count
(and hence the number of visual tokens, before the processor's own rounding to
multiples of 32 px and its min/max pixel clamp) shrinks by about `keep`. Unlike
pruning after the merger, this also removes ViT compute.
"""
import math

import numpy as np
from PIL import Image


def downscale(image, keep):
    if keep >= 1.0:
        return image
    s = math.sqrt(keep)
    if isinstance(image, Image.Image):
        w, h = image.size
        return image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    pil = Image.fromarray(arr.astype(np.uint8))
    return np.asarray(downscale(pil, keep))
