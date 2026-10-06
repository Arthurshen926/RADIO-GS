"""Real native encoders and all four source-free query adapters; no accuracy claim."""
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image
import torch

from .adapters import NativeQueryEncoder
from .cold_check import outputs
from .storage import load_field, sha256


def smoke(root, device="cuda"):
    root = Path(root).resolve()
    report_path = root / "runtime_smoke.json"
    if report_path.exists():
        raise FileExistsError("runtime smoke report already exists")
    observations = root / "observations"
    manifest = json.loads((observations / "observations.json").read_text())
    entry = next(row for row in manifest["frames"] if row["split"] == "development")
    image_path = Path(entry["rgb"]["path"])
    if sha256(image_path) != entry["rgb"]["sha256"]:
        raise ValueError("query-image fixture changed")
    view = torch.load(observations / f"view_{entry['frame']:05d}.pt", map_location="cpu", weights_only=True)
    started = time.monotonic()
    encoder = NativeQueryEncoder(manifest["native_semantic_space"], device)
    text = encoder.text("a figurine")
    image = encoder.image(Image.open(image_path).convert("RGB"))
    fixture = dict(text=text.semantic.cpu(), image_semantic=image.semantic.cpu(), image_local=image.local.cpu())
    encoder_receipt = dict(siglip_bundle=encoder._vision.bundle,
                          dino_checkpoint=manifest["dino_checkpoint"],
                          native_encoder_load_and_two_inputs_seconds=time.monotonic()-started)
    del encoder, text, image
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    field, export = load_field(root / "models" / "trained", device)
    height, width = manifest["raster_hw"]
    # The fixture chooses a supported camera ray by geometry alone, never by
    # teacher labels, query score, membership output or benchmark answers.
    operator = view["operator"].coalesce()
    known_mass = torch.sparse.mm(operator, field.observed.cpu().float()[:, None])[:, 0]
    candidates = torch.where((known_mass > 0) & (view["best_element"] >= 0))[0]
    candidates = candidates[field.observed.cpu()[view["best_element"][candidates]]]
    if not len(candidates):
        raise ValueError("smoke camera has no registered observed support")
    centre_distance = (candidates//width-height/2)**2+(candidates%width-width/2)**2
    pixel = int(candidates[centre_distance.argmin()])
    k = view["intrinsic"].clone()
    k[0] *= width/view["original_hw"][1]
    k[1] *= height/view["original_hw"][0]
    fixture.update(pixel_xy=[pixel%width, pixel//width], xyz=field.xyz[int(view["best_element"][pixel])].cpu(),
                   camera=dict(intrinsic=k, world_to_camera=view["world_to_camera"], hw=[height,width],
                               world_units_per_metre=manifest["world_units_per_metre"]))
    expected = outputs(field, fixture)
    del field
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    repository = Path(__file__).resolve().parents[2]
    forbidden = [repository / "output", Path("/root/RADIO-GS/output"), image_path.parents[1]]
    # Copy bytes into a new filesystem location. The child starts outside the
    # repository, receives no observation paths and forbids source-file opens.
    with tempfile.TemporaryDirectory(prefix="query-field-cold-") as directory:
        isolated = Path(directory)
        shutil.copytree(root / "models" / "trained", isolated / "scene")
        torch.save(fixture, isolated / "query.pt")
        command = [sys.executable, "-m", "radio_gs.query_field.cold_check", "--field", str(isolated / "scene"),
                   "--fixture", str(isolated / "query.pt"), "--device", device]
        for path in forbidden:
            command.extend(["--deny", str(path.resolve())])
        environment = dict(os.environ, PYTHONPATH=str(repository))
        process = subprocess.run(command, cwd=isolated, env=environment, capture_output=True, text=True, check=True)
        child = json.loads(process.stdout.strip().splitlines()[-1])
    parity = {name: all(expected[name][key] == child["results"][name][key]
                       for key in ["scores_sha256", "selected_sha256", "selected"]) for name in expected}
    if not all(parity.values()) or child["forbidden_open_attempts"]:
        raise RuntimeError("fresh-process query replay differs or accessed sources")
    report = dict(schema="radio_gs.query_field_runtime_smoke.v1", benchmark_result=False,
        field_sha256=export["files"]["field.pt"]["sha256"], copied_export=True, fresh_process=True,
        query_fixture=dict(text="a figurine", image=entry["rgb"], image_scope="whole authorized current query image",
                           camera_frame=entry["frame"], pixel_xy=fixture["pixel_xy"],
                           point_selection="nearest centre camera ray with observed Gaussian association; geometry-only fixture"),
        encoder=encoder_receipt, exact_replay=parity, results=child["results"],
        forbidden_open_attempts=child["forbidden_open_attempts"], enforcement=child["enforcement"],
        external_encoders_are_shared_runtime_dependencies=True,
        semantic_increment_bytes=export["semantic_increment_bytes"],
        limitations="Functional query/adaptor and source-denial check only. Whole-image input does not provide a unique object target; no text/image IoU is claimed.")
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    return report
