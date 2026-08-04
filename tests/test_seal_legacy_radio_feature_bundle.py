import hashlib
import json
from pathlib import Path

import pytest
import torch

from radio_gs.scripts.extract_radio_features import (
    LEGACY_RESEAL_CONTRACT,
    _sha256_file,
    _validate_final_output_bundle,
)
from radio_gs.scripts.seal_legacy_radio_feature_bundle import (
    seal_legacy_bundle,
)


def _legacy_bundle(tmp_path: Path) -> tuple[Path, bytes, list[Path]]:
    root = tmp_path / "features"
    images = tmp_path / "images"
    (root / "backbone").mkdir(parents=True)
    (root / "summary").mkdir()
    images.mkdir()
    frames = []
    tensors = []
    for index in range(2):
        source = images / f"frame_{index:05d}.jpg"
        source.write_bytes(f"image-{index}".encode("ascii"))
        stem = f"rgb_{index}"
        backbone = root / "backbone" / f"{stem}.pt"
        summary = root / "summary" / f"{stem}.pt"
        torch.save(torch.full((3, 2, 2), index + 1, dtype=torch.float16), backbone)
        torch.save(torch.full((5,), index + 1, dtype=torch.float32), summary)
        tensors.extend([backbone, summary])
        frames.append(
            {
                "source_rank": index,
                "frame_idx": index,
                "source_file": source.name,
                "source_sha256": _sha256_file(source),
                "saved_stem": stem,
            }
        )
    manifest = {
        "scene": "test",
        "radio": {
            "version": "c-radio_v4-h",
            "repo": "/legacy/RADIO",
            "requested_adaptors": [],
        },
        "image_dir": str(images),
        "num_frames": 2,
        "frames": frames,
        "features": {
            "backbone": {
                "subdir": "backbone",
                "dim": 3,
                "grid": [2, 2],
                "dtype": "float16",
            },
            "summary": {
                "subdir": "summary",
                "dim": 5,
                "dtype": "float32",
            },
            "adaptors": [],
        },
    }
    raw = json.dumps(manifest, indent=2).encode("utf-8") + b"\n"
    (root / "frame_manifest.json").write_bytes(raw)
    return root, raw, tensors


def test_legacy_reseal_is_hash_bound_and_does_not_modify_tensors(tmp_path: Path) -> None:
    root, raw_manifest, tensors = _legacy_bundle(tmp_path)
    legacy_sha = hashlib.sha256(raw_manifest).hexdigest()
    tensor_hashes = {path: _sha256_file(path) for path in tensors}

    with pytest.raises(ValueError, match="SHA-256"):
        seal_legacy_bundle(
            root,
            expected_legacy_manifest_sha256="0" * 64,
        )
    assert (root / "frame_manifest.json").read_bytes() == raw_manifest

    receipt = seal_legacy_bundle(
        root,
        expected_legacy_manifest_sha256=legacy_sha,
    )
    sealed = json.loads((root / "frame_manifest.json").read_text())
    assert sealed["execution"]["formalization_contract"] == LEGACY_RESEAL_CONTRACT
    assert (root / "frame_manifest.legacy.json").read_bytes() == raw_manifest
    assert receipt["num_frames"] == 2
    assert receipt["num_tensors"] == 4
    assert all(_sha256_file(path) == digest for path, digest in tensor_hashes.items())

    validation = _validate_final_output_bundle(
        root,
        expected_output_bundle_sha256=receipt["output_bundle_sha256"],
    )
    assert validation["legacy_source_manifest_sha256"] == legacy_sha
    assert validation["num_frames"] == 2


def test_legacy_reseal_validator_rejects_tensor_change(tmp_path: Path) -> None:
    root, raw_manifest, tensors = _legacy_bundle(tmp_path)
    receipt = seal_legacy_bundle(
        root,
        expected_legacy_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
    )
    torch.save(torch.zeros(3, 2, 2, dtype=torch.float16), tensors[0])

    with pytest.raises(ValueError, match="cannot be reopened"):
        _validate_final_output_bundle(
            root,
            expected_output_bundle_sha256=receipt["output_bundle_sha256"],
        )
