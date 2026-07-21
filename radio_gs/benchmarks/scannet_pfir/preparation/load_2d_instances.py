"""Read official projected 2-D instance masks without decoding them as 3-D IDs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_2d_instances(path: str | Path) -> dict[int, np.ndarray]:
    values = np.asarray(Image.open(path), dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("ScanNet instance projection must be one-channel")
    return {
        int(instance_id): values == instance_id
        for instance_id in np.unique(values)
        if int(instance_id) > 0
    }

