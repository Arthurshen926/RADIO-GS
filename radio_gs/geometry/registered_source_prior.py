"""Camera-only adapter for registered, metre-scale, provided-mesh inputs.

Implements the camera/scale interface used by source rendering stages. It never
fabricates MoGe predictions, depth calibration evidence or COLMAP tracks.
"""
import json
from pathlib import Path
from radio_gs.geometry.registered_pinhole import load_registered_camera_table
from radio_gs.utils.immutable_artifacts import sha256_file


class RegisteredSourcePrior:
    def __init__(self, path, expected_sha256):
        if sha256_file(path) != expected_sha256:
            raise ValueError('Registered source camera authority changed')
        data = json.loads(Path(path).read_text())
        self.cameras = load_registered_camera_table(path)
        self.scale = 1.
        self.manifest = {'scene_label': data['scene']}
        self.records = {int(r['frame_index']): dict(source_image=r['source_image'],
            source_image_sha256=r['source_sha256']) for r in data['records']}
