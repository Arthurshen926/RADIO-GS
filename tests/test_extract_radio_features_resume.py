from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from radio_gs.scripts import extract_radio_features as extraction


def _write_pickle_sentinel(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousResumePayload:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        return _write_pickle_sentinel, (str(self.sentinel),)


def _images(tmp_path: Path, count: int) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    for index in range(count):
        Image.new(
            "RGB",
            (32, 32),
            color=(index + 1, index + 2, index + 3),
        ).save(image_dir / f"rgb_{index}.png")
    radio_repo = tmp_path / "RADIO"
    radio_repo.mkdir(exist_ok=True)
    (radio_repo / "hubconf.py").write_text(
        "from implementation import radio_model\n",
        encoding="utf-8",
    )
    (radio_repo / "implementation.py").write_text(
        "GENERATION = 1\n",
        encoding="utf-8",
    )
    return image_dir


def _args(
    image_dir: Path,
    output_dir: Path,
    *,
    resume_partial: bool,
    pacing_seconds: float = 0.0,
    device: str = "cpu",
) -> Namespace:
    return Namespace(
        scene="toy",
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        radio_repo=str(image_dir.parent / "RADIO"),
        radio_version="toy-radio",
        radio_checkpoint="",
        batch_size=1,
        frame_stride=1,
        max_frames=None,
        frame_id_mode="auto",
        exclude_image_stem=[],
        exclude_image_stems_file="",
        extract_adaptors=True,
        adaptor_names="dino_v3_7b,sam3",
        resolution_scale=1.0,
        sliding_window=False,
        tile_size=1024,
        tile_overlap=128,
        device=device,
        amp=False,
        skip_pca_stats=True,
        resume_partial=resume_partial,
        radio_thermal_pacing_seconds_per_image=pacing_seconds,
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool = False,
) -> dict[str, int]:
    calls = {"model_load": 0, "forward": 0}

    def fake_model_load(*_args, **_kwargs):
        calls["model_load"] += 1
        return object(), object()

    def fake_preprocess(paths, _target_h, _target_w, _device):
        frame_ids = [int(path.stem.rsplit("_", 1)[-1]) for path in paths]
        return torch.tensor(frame_ids, dtype=torch.float32).reshape(-1, 1, 1, 1)

    def fake_forward(
        _model,
        _conditioner,
        imgs,
        _amp,
        _patch_h,
        _patch_w,
        adaptor_names=None,
    ):
        calls["forward"] += 1
        frame_ids = imgs[:, 0, 0, 0]
        batch = int(frame_ids.shape[0])
        backbone = frame_ids.reshape(batch, 1, 1, 1) + torch.arange(
            8,
            dtype=torch.float32,
        ).reshape(1, 2, 2, 2)
        summary = torch.stack((frame_ids, frame_ids + 0.5), dim=1)
        adaptors = {
            "dino_v3_7b": frame_ids.reshape(batch, 1, 1, 1)
            + torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2),
            "sam3": frame_ids.reshape(batch, 1, 1, 1)
            + torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2),
        }
        assert list(adaptor_names or []) == ["dino_v3_7b", "sam3"]
        return summary, backbone, adaptors

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    if cuda_available:
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _device: SimpleNamespace(
                name="mock-cuda",
                major=8,
                minor=6,
                total_memory=24 * 1024**3,
            ),
        )
    monkeypatch.setattr(extraction, "_load_radio_model", fake_model_load)
    monkeypatch.setattr(extraction, "_load_and_preprocess", fake_preprocess)
    monkeypatch.setattr(extraction, "_run_radio_batch", fake_forward)
    return calls


def _saved_tensors(root: Path) -> dict[str, torch.Tensor]:
    result = {}
    for subdir in ("backbone", "summary", "dino_v3_7b", "sam3"):
        for path in sorted((root / subdir).glob("*.pt")):
            result[str(path.relative_to(root))] = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
    return result


def _assert_tensors_exactly_equal(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
) -> None:
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].dtype == second[key].dtype
        assert first[key].shape == second[key].shape
        assert torch.equal(first[key], second[key]), key


class _FakeHubRadioModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def make_preprocessor_external(self):
        return "conditioner"


def test_explicit_radio_checkpoint_is_restricted_before_hub_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "radio.pt"
    torch.save(
        {
            "state_dict": {"weight": torch.tensor([1.25])},
            "args": argparse.Namespace(arch="test-radio"),
        },
        checkpoint,
    )
    expected_sha256 = extraction._sha256_file(checkpoint)
    observed = {}
    original_torch_load = torch.load

    def fake_hub_load(_repo, entrypoint, *, version, **_kwargs):
        assert entrypoint == "radio_model"
        observed["payload"] = torch.load(
            version,
            map_location="cpu",
            weights_only=False,
        )
        return _FakeHubRadioModel()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    model, conditioner = extraction._load_radio_model(
        str(tmp_path),
        str(checkpoint),
        ["sam3"],
        torch.device("cpu"),
        expected_checkpoint_sha256=expected_sha256,
    )

    assert isinstance(model, _FakeHubRadioModel)
    assert conditioner == "conditioner"
    assert observed["payload"]["args"].arch == "test-radio"
    assert torch.load is original_torch_load


def test_explicit_radio_checkpoint_rejects_wrong_authority_before_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "radio.pt"
    torch.save({"state_dict": {}}, checkpoint)
    called = False

    def fake_hub_load(*_args, **_kwargs):
        nonlocal called
        called = True
        return _FakeHubRadioModel()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        extraction._load_radio_model(
            str(tmp_path),
            str(checkpoint),
            None,
            torch.device("cpu"),
            expected_checkpoint_sha256="0" * 64,
        )

    assert called is False


def test_formal_radio_hub_rejects_any_second_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "radio.pt"
    unrelated = tmp_path / "unrelated.pt"
    torch.save({"state_dict": {}}, checkpoint)
    torch.save(torch.ones(1), unrelated)
    expected_sha256 = extraction._sha256_file(checkpoint)
    original_torch_load = torch.load

    def fake_hub_load(_repo, _entrypoint, *, version, **_kwargs):
        torch.load(version, map_location="cpu", weights_only=False)
        torch.load(unrelated, map_location="cpu", weights_only=False)
        return _FakeHubRadioModel()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    with pytest.raises(RuntimeError, match="unapproved torch.load"):
        extraction._load_radio_model(
            str(tmp_path),
            str(checkpoint),
            None,
            torch.device("cpu"),
            expected_checkpoint_sha256=expected_sha256,
        )

    assert torch.load is original_torch_load


def test_formal_radio_hub_rejects_checkpoint_code_before_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = tmp_path / "must_not_exist"
    checkpoint = tmp_path / "malicious_radio.pt"
    torch.save({"payload": _MaliciousResumePayload(sentinel)}, checkpoint)
    called = False

    def fake_hub_load(*_args, **_kwargs):
        nonlocal called
        called = True
        return _FakeHubRadioModel()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    with pytest.raises(Exception, match="forbidden RADIO checkpoint global"):
        extraction._load_radio_model(
            str(tmp_path),
            str(checkpoint),
            None,
            torch.device("cpu"),
            expected_checkpoint_sha256=extraction._sha256_file(checkpoint),
        )

    assert called is False
    assert not sentinel.exists()


def test_pacing_does_not_change_any_scientific_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 3)
    _install_fake_runtime(monkeypatch)
    sleeps: list[float] = []
    monkeypatch.setattr(extraction.time, "sleep", sleeps.append)

    unpaced = tmp_path / "unpaced"
    paced = tmp_path / "paced"
    extraction.extract(_args(image_dir, unpaced, resume_partial=False))
    extraction.extract(
        _args(
            image_dir,
            paced,
            resume_partial=False,
            pacing_seconds=2.5,
        )
    )

    _assert_tensors_exactly_equal(
        _saved_tensors(unpaced),
        _saved_tensors(paced),
    )
    assert sleeps == [2.5, 2.5, 2.5]
    execution = json.loads((paced / "frame_manifest.json").read_text())["execution"]
    assert execution["radio_thermal_pacing_seconds_per_image"] == 2.5
    assert execution["pacing_order"] == (
        "frame_commit_then_cuda_synchronize_then_sleep_v1"
    )


def test_interrupted_resume_matches_clean_extraction_and_skips_committed_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 4)
    calls = _install_fake_runtime(monkeypatch)
    partial = tmp_path / "partial"
    clean = tmp_path / "clean"
    pace_calls = 0

    def interrupt_after_second_commit(_device, _seconds):
        nonlocal pace_calls
        pace_calls += 1
        if pace_calls == 2:
            raise RuntimeError("injected interruption")

    monkeypatch.setattr(extraction, "_thermal_pause", interrupt_after_second_commit)
    with pytest.raises(RuntimeError, match="injected interruption"):
        extraction.extract(_args(image_dir, partial, resume_partial=True))
    assert not (partial / "frame_manifest.json").exists()
    assert len(list((partial / extraction.FRAME_COMMIT_DIRNAME).glob("*.json"))) == 2

    monkeypatch.setattr(extraction, "_thermal_pause", lambda *_args: None)
    before_resume = calls["forward"]
    extraction.extract(_args(image_dir, partial, resume_partial=True))
    assert calls["forward"] - before_resume == 2

    before_clean = calls["forward"]
    extraction.extract(_args(image_dir, clean, resume_partial=True))
    assert calls["forward"] - before_clean == 4
    _assert_tensors_exactly_equal(
        _saved_tensors(partial),
        _saved_tensors(clean),
    )
    assert (partial / "frame_manifest.json").read_bytes() == (
        clean / "frame_manifest.json"
    ).read_bytes()


def test_bad_or_missing_committed_tensor_recomputes_entire_affected_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 3)
    calls = _install_fake_runtime(monkeypatch)
    output = tmp_path / "features"
    args = _args(image_dir, output, resume_partial=True)
    extraction.extract(args)
    expected = _saved_tensors(output)

    (output / "backbone" / "rgb_0.pt").write_bytes(b"damaged")
    (output / "summary" / "rgb_1.pt").unlink()
    before_resume = calls["forward"]
    extraction.extract(args)

    assert calls["forward"] - before_resume == 2
    _assert_tensors_exactly_equal(expected, _saved_tensors(output))


def test_final_bundle_rejects_coordinated_tensor_and_marker_tamper_then_repairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 2)
    calls = _install_fake_runtime(monkeypatch)
    output = tmp_path / "features"
    args = _args(image_dir, output, resume_partial=True)
    extraction.extract(args)
    original_manifest = (output / "frame_manifest.json").read_bytes()
    target = output / "dino_v3_7b" / "rgb_0.pt"
    value = torch.load(target, map_location="cpu", weights_only=True)
    extraction._atomic_torch_save(value + 100, target)
    marker_path = (
        output / extraction.FRAME_COMMIT_DIRNAME / "rgb_0.json"
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for record in marker["tensors"]:
        if record["relative_path"] == "dino_v3_7b/rgb_0.pt":
            record["sha256"] = extraction._sha256_file(target)
    extraction._atomic_json_write(marker_path, marker)

    with pytest.raises(ValueError, match="final feature frame"):
        extraction._validate_final_output_bundle(output)

    before = dict(calls)
    extraction.extract(args)

    assert calls["model_load"] - before["model_load"] == 1
    assert calls["forward"] - before["forward"] == 1
    repaired = torch.load(target, map_location="cpu", weights_only=True)
    assert torch.equal(repaired, value)
    assert (output / "frame_manifest.json").read_bytes() == original_manifest
    validation = extraction._validate_final_output_bundle(output)
    manifest = json.loads((output / "frame_manifest.json").read_text())
    assert validation["output_bundle_sha256"] == manifest["output_bundle_sha256"]
    assert len(manifest["output_bundle"]["frames"]) == 2
    runtime = dict(manifest["execution"]["runtime_fingerprint"])
    runtime_sha = runtime.pop("fingerprint_sha256")
    assert runtime_sha == extraction._canonical_json_sha256(runtime)
    source_files = manifest["radio"]["python_source_tree"]["files"]
    assert [record["relative_path"] for record in source_files] == sorted(
        record["relative_path"] for record in source_files
    )


def test_radio_imported_source_change_rejects_partial_resume_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 2)
    calls = _install_fake_runtime(monkeypatch)
    output = tmp_path / "features"
    args = _args(image_dir, output, resume_partial=True)

    def interrupt_after_first(*_args):
        raise RuntimeError("injected interruption")

    monkeypatch.setattr(extraction, "_thermal_pause", interrupt_after_first)
    with pytest.raises(RuntimeError, match="injected interruption"):
        extraction.extract(args)
    (tmp_path / "RADIO" / "implementation.py").write_text(
        "GENERATION = 2\n",
        encoding="utf-8",
    )
    before = calls["model_load"]
    monkeypatch.setattr(extraction, "_thermal_pause", lambda *_args: None)

    with pytest.raises(ValueError, match="resume contract differs"):
        extraction.extract(args)

    assert calls["model_load"] == before


def test_all_committed_bundle_skips_model_and_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 2)
    calls = _install_fake_runtime(monkeypatch)
    output = tmp_path / "features"
    args = _args(image_dir, output, resume_partial=True)
    extraction.extract(args)
    before = dict(calls)

    extraction.extract(args)

    assert calls == before


def test_resume_contract_change_fails_closed_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 2)
    calls = _install_fake_runtime(monkeypatch)
    output = tmp_path / "features"
    extraction.extract(_args(image_dir, output, resume_partial=True))
    changed = _args(
        image_dir,
        output,
        resume_partial=True,
        pacing_seconds=1.0,
    )
    before = calls["model_load"]

    with pytest.raises(ValueError, match="resume contract differs"):
        extraction.extract(changed)

    assert calls["model_load"] == before


def test_uncontracted_partial_directory_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 1)
    calls = _install_fake_runtime(monkeypatch)
    output = tmp_path / "features"
    output.mkdir()
    (output / "unknown-partial-file").write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError, match="has no resume contract"):
        extraction.extract(_args(image_dir, output, resume_partial=True))

    assert calls["model_load"] == 0


def test_atomic_writers_preserve_existing_target_on_failed_temp_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor_path = tmp_path / "rgb_0.pt"
    torch.save(torch.tensor([1.0]), tensor_path)
    original_tensor_bytes = tensor_path.read_bytes()

    def failed_torch_save(_value, temporary_handle):
        temporary_handle.write(b"partial")
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(extraction.torch, "save", failed_torch_save)
    with pytest.raises(RuntimeError, match="injected save failure"):
        extraction._atomic_torch_save(torch.tensor([2.0]), tensor_path)
    assert tensor_path.read_bytes() == original_tensor_bytes
    assert list(tmp_path.glob(".*.tmp")) == []

    manifest_path = tmp_path / "frame_manifest.json"
    manifest_path.write_text('{"old": true}\n', encoding="utf-8")
    original_manifest_bytes = manifest_path.read_bytes()
    with pytest.raises(TypeError):
        extraction._atomic_json_write(manifest_path, {"invalid": object()})
    assert manifest_path.read_bytes() == original_manifest_bytes
    assert list(tmp_path.glob(".*.tmp")) == []


def test_resume_tensor_reopen_uses_weights_only_and_never_executes_pickle(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "pickle-executed"
    artifact = tmp_path / "malicious.pt"
    torch.save(_MaliciousResumePayload(sentinel), artifact)
    record = {
        "sha256": extraction._sha256_file(artifact),
        "dtype": "float16",
        "shape": [1],
        "num_bytes": 2,
    }

    with pytest.raises(ValueError, match="cannot be reopened"):
        extraction._load_validated_tensor(artifact, record)

    assert not sentinel.exists()


def test_commit_precedes_cuda_synchronize_and_sleep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = _images(tmp_path, 1)
    _install_fake_runtime(monkeypatch, cuda_available=True)
    events: list[str] = []
    original_atomic_json_write = extraction._atomic_json_write

    def recording_json_write(path, payload):
        original_atomic_json_write(path, payload)
        if Path(path).parent.name == extraction.FRAME_COMMIT_DIRNAME:
            events.append("commit")

    monkeypatch.setattr(extraction, "_atomic_json_write", recording_json_write)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: events.append("synchronize"),
    )
    monkeypatch.setattr(extraction.time, "sleep", lambda _value: events.append("sleep"))

    extraction.extract(
        _args(
            image_dir,
            tmp_path / "features",
            resume_partial=True,
            pacing_seconds=1.0,
            device="cuda:0",
        )
    )

    assert events[:3] == ["commit", "synchronize", "sleep"]


@pytest.mark.parametrize(
    ("batch_size", "skip_pca_stats", "message"),
    [
        (2, True, "requires --batch_size 1"),
        (1, False, "requires --skip_pca_stats"),
    ],
)
def test_resume_rejects_unsafe_batch_or_pca_mode(
    tmp_path: Path,
    batch_size: int,
    skip_pca_stats: bool,
    message: str,
) -> None:
    image_dir = _images(tmp_path, 1)
    args = _args(image_dir, tmp_path / "features", resume_partial=True)
    args.batch_size = batch_size
    args.skip_pca_stats = skip_pca_stats

    with pytest.raises(ValueError, match=message):
        extraction.extract(args)


def test_resume_rejects_nonfinite_or_negative_pacing(tmp_path: Path) -> None:
    image_dir = _images(tmp_path, 1)
    for value in (float("nan"), float("inf"), -0.1):
        args = _args(
            image_dir,
            tmp_path / str(value),
            resume_partial=True,
            pacing_seconds=value,
        )
        with pytest.raises(ValueError, match="finite and non-negative"):
            extraction.extract(args)
