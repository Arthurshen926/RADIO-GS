"""Optional bounded sink for the unchanged public prediction payload formats."""
import json
from pathlib import Path

import numpy as np
import torch


def persist(path, value, kind, writer=None):
    if writer is not None:
        return writer.public_artifact(path, value, kind)
    if kind == "tensor":
        torch.save(value, path)
    elif kind == "numpy":
        np.save(path, value)
    elif kind == "json":
        Path(path).write_text(json.dumps(value, indent=2) + "\n")
    else:
        raise ValueError("unknown public artifact type")
