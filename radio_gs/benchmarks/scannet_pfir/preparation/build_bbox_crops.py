"""Reproducible BBox and masked-oracle crops."""

from __future__ import annotations

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_pfir.protocol import padded_bbox


def build_bbox_crop(
    image: Image.Image, mask: np.ndarray, *, padding: float = 0.10, masked: bool = False
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    rgb = image.convert("RGB")
    box = padded_bbox(mask, padding)
    if not masked:
        return rgb.crop(box), box
    array = np.asarray(rgb, dtype=np.uint8)
    values = np.asarray(mask, dtype=bool)
    if values.shape != array.shape[:2]:
        raise ValueError("mask/image shapes differ")
    masked_rgb = np.where(values[..., None], array, 0)
    x0, y0, x1, y1 = box
    return Image.fromarray(masked_rgb[y0:y1, x0:x1]), box

