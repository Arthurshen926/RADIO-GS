"""Load exactly the RGB crop exposed by the method-input manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


def load_method_crop(record: Mapping) -> np.ndarray:
    allowed = set(record.get("available_method_inputs", ()))
    if allowed != {"scene_id", "crop_rgb"}:
        raise ValueError("record does not satisfy the PFIR method-input contract")
    path = Path(record["crop_rgb_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

