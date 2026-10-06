"""Fresh-process source-denial check, using only a local export and query packet."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from .adapters import point_2d, point_3d
from .contracts import QueryPacket
from .storage import load_field


def outputs(field, fixture):
    camera = fixture["camera"]
    packets = {
        "text": QueryPacket("text", semantic=fixture["text"]),
        "image": QueryPacket("image", semantic=fixture["image_semantic"], local=fixture["image_local"]),
        "point_2d": point_2d(field, fixture["pixel_xy"], camera["intrinsic"], camera["world_to_camera"],
                             camera["hw"], world_units_per_metre=camera["world_units_per_metre"]),
        "point_3d": point_3d(field, fixture["xyz"]),
    }
    result = {}
    for name, packet in packets.items():
        started = time.monotonic()
        queried = field.query(packet)
        values = queried.membership_scores.cpu().numpy().tobytes()
        selected = queried.selected.cpu().numpy().tobytes()
        result[name] = dict(scores_sha256=hashlib.sha256(values).hexdigest(),
            selected_sha256=hashlib.sha256(selected).hexdigest(), selected=int(queried.selected.sum()),
            seconds=time.monotonic()-started, diagnostics=queried.diagnostics)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--deny", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    forbidden = [Path(path).resolve() for path in args.deny]
    attempts = []
    def guard(event, values):
        if event != "open" or not isinstance(values[0], (str, bytes)):
            return
        path = Path(values[0].decode() if isinstance(values[0], bytes) else values[0]).resolve()
        if any(path == prefix or prefix in path.parents for prefix in forbidden):
            attempts.append(str(path))
            raise PermissionError("source dependency access denied: " + str(path))
    sys.addaudithook(guard)
    field, _ = load_field(args.field, args.device)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    result = outputs(field, fixture)
    print(json.dumps(dict(results=result, forbidden_open_attempts=attempts,
                         enforcement="Python file-open audit; not an OS security sandbox")))


if __name__ == "__main__":
    main()
