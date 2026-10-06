"""Read selected ScanNet RGB-D observations without loading an entire scan."""
from __future__ import annotations

import io
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image


class SensorIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.frames = []
        with self.path.open("rb") as f:
            if struct.unpack("I", f.read(4))[0] != 4:
                raise ValueError("Expected ScanNet .sens version 4")
            n = struct.unpack("Q", f.read(8))[0]
            f.seek(n, 1)
            self.k_color = np.frombuffer(f.read(64), "f4").reshape(4, 4).copy()
            e_color = np.frombuffer(f.read(64), "f4").reshape(4, 4)
            self.k_depth = np.frombuffer(f.read(64), "f4").reshape(4, 4).copy()
            e_depth = np.frombuffer(f.read(64), "f4").reshape(4, 4)
            # All nine bound scans have coincident registered camera coordinates.
            if not (np.allclose(e_color, np.eye(4)) and np.allclose(e_depth, np.eye(4))):
                raise ValueError("This construction track requires registered ScanNet cameras")
            self.color_compression, self.depth_compression = struct.unpack("ii", f.read(8))
            self.color_width, self.color_height, self.width, self.height = struct.unpack("IIII", f.read(16))
            self.depth_scale = struct.unpack("f", f.read(4))[0]
            count = struct.unpack("Q", f.read(8))[0]
            for _ in range(count):
                pose = np.frombuffer(f.read(64), "f4").reshape(4, 4).copy()
                f.seek(16, 1)
                color_n, depth_n = struct.unpack("QQ", f.read(16))
                offset = f.tell()
                self.frames.append((pose, offset, color_n, depth_n))
                f.seek(color_n + depth_n, 1)

    def read(self, index: int, *, color: bool = True):
        pose, offset, color_n, depth_n = self.frames[index]
        with self.path.open("rb") as f:
            f.seek(offset)
            color_bytes = f.read(color_n) if color else None
            f.seek(offset + color_n)
            raw = f.read(depth_n)
        if self.depth_compression == 1:
            raw = zlib.decompress(raw)
        elif self.depth_compression != 0:
            raise ValueError("Unsupported ScanNet depth compression")
        depth = np.frombuffer(raw, "u2").reshape(self.height, self.width).astype(np.float32) / self.depth_scale
        rgb = Image.open(io.BytesIO(color_bytes)).convert("RGB") if color else None
        return pose.astype(np.float64), depth, rgb

    def exclude(self, query_indices: list[int]):
        poses = np.asarray([f[0] for f in self.frames], dtype=np.float64)
        finite = np.isfinite(poses).all(axis=(1, 2))
        withheld = np.zeros(len(poses), dtype=bool)
        for q in query_indices:
            withheld[max(0, q - 5):q + 6] = True
            delta = np.linalg.norm(poses[:, :3, 3] - poses[q, :3, 3], axis=1)
            relative = np.einsum("ij,njk->nik", poses[q, :3, :3].T, poses[:, :3, :3])
            angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
            withheld |= finite & (delta < .10) & (angle < 8.)
        return np.flatnonzero(finite & ~withheld), np.flatnonzero(withheld)
