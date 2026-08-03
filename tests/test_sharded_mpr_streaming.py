from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_saved_responsibility_for_sharded_resume,
    _stream_channel_sharded_contribution_mean,
    accumulate_contribution_mean_channel_chunked,
    finalize_registered_mean_chunked,
)
from radio_gs.scripts.train_canonical_radio_field import (
    _basis_fit_values,
    train,
)
from radio_gs.training.canonical_field_losses import _capability_consensus_loss
from radio_gs.training.primitive_consensus import (
    PrimitiveConsensus,
    primitive_reconstruction_loss,
)
from radio_gs.training.tensor_cache_io import ShardedMPRCache, load_mpr_cache
from radio_gs.utils.immutable_artifacts import sha256_file


def _xyz_sha256(xyz: torch.Tensor) -> str:
    return hashlib.sha256(
        xyz.float().contiguous().numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()


def _write_sharded_cache(
    root: Path,
    features: torch.Tensor,
    *,
    counts: torch.Tensor | None = None,
    shard_channels: int = 2,
    feature_space: str = "radio",
) -> Path:
    values = features.half().contiguous()
    rows, channels = values.shape
    xyz = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    counts = (
        torch.tensor([2, 1, 0, 2], dtype=torch.long)[:rows]
        if counts is None
        else counts.long()
    )
    valid = counts > 0
    values[~valid] = 0
    reliability = torch.stack(
        [counts.float() / 2.0, valid.float(), valid.float()], dim=-1
    ).half()
    support = root / "target.support.pt"
    torch.save(
        {
            "xyz": xyz,
            "valid": valid,
            "view_counts": counts,
            "reliability": reliability,
        },
        support,
    )
    shards = []
    for start in range(0, channels, shard_channels):
        stop = min(start + shard_channels, channels)
        path = root / f"target.channels_{start}_{stop}.f16"
        path.write_bytes(
            values[:, start:stop]
            .numpy()
            .astype("<f2", copy=False)
            .tobytes(order="C")
        )
        shards.append(
            {
                "relative_path": path.name,
                "sha256": sha256_file(path),
                "channel_start": start,
                "channel_stop": stop,
                "dtype": "float16",
                "shape": [rows, stop - start],
            }
        )
    xyz_digest = _xyz_sha256(xyz)
    metadata = {
        "schema_version": 1,
        "feature_space": feature_space,
        "num_declared_views": 2,
        "selected_frame_indices": [3, 7],
        "xyz_sha256": xyz_digest,
        "raster_reliability_mode": "legacy_valid",
        "aggregation_mode": "raster_gaussian_top1",
        "shared_registration_responsibility": True,
        "registration_responsibility_cache_sha256": "c" * 64,
        "feature_output_bundle_sha256": "b" * 64,
        "benchmark_masks_opened": False,
        "benchmark_images_opened": False,
        "text_queries_opened": False,
    }
    manifest = root / "target.pt"
    manifest.write_text(
        json.dumps(
            {
                "schema": "radio_gs.channel_sharded_mpr.v1",
                "schema_version": 1,
                "layout": "row_major_channel_shards",
                "feature_dtype": "float16",
                "feature_shape": [rows, channels],
                "support": {
                    "relative_path": support.name,
                    "sha256": sha256_file(support),
                },
                "shards": shards,
                "geometry_fingerprint": {
                    "num_gaussians": rows,
                    "xyz_sha256": xyz_digest,
                },
                "metadata": metadata,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def _copy_sharded_members_to_mirror(manifest: Path, mirror: Path) -> list[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    names = [
        str(payload["support"]["relative_path"]),
        *(str(record["relative_path"]) for record in payload["shards"]),
    ]
    mirror.mkdir()
    for name in names:
        shutil.copyfile(manifest.parent / name, mirror / name)
    return names


def test_sharded_cache_random_row_fetch_matches_dense(tmp_path: Path) -> None:
    dense = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [-1.0, 0.5, 8.0, 7.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [9.0, 3.0, 1.0, 4.0, 6.0],
        ]
    ).half()
    manifest = _write_sharded_cache(tmp_path, dense)
    cache, digest, source = load_mpr_cache(
        manifest,
        expected_sha256=sha256_file(manifest),
        expected_feature_space="radio",
        require_formal_safety=True,
    )
    assert isinstance(cache, ShardedMPRCache)
    assert digest == sha256_file(manifest)
    assert source == manifest.resolve()
    rows = torch.tensor([3, 0, 3, 1])
    torch.testing.assert_close(cache.fetch_rows(rows), dense[rows], rtol=0, atol=0)
    assert len(cache.provenance()["shards"]) == 3


def test_sharded_cache_without_mirror_env_reads_authority_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RADIO_GS_MPR_LOCAL_MIRROR_DIR", raising=False)
    dense = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [7.0, 8.0, 9.0]]
    ).half()
    manifest = _write_sharded_cache(tmp_path, dense)

    cache, _digest, _source = load_mpr_cache(manifest)

    assert isinstance(cache, ShardedMPRCache)
    assert all(shard.mapped_path == shard.path for shard in cache.shards)
    torch.testing.assert_close(
        cache.fetch_rows(torch.tensor([3, 0])), dense[[3, 0]], rtol=0, atol=0
    )


def test_sharded_cache_reads_validated_local_mirror_without_changing_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    dense = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [7.0, 8.0, 9.0]]
    ).half()
    manifest = _write_sharded_cache(authority, dense)
    mirror = tmp_path / "mirror"
    _copy_sharded_members_to_mirror(manifest, mirror)
    monkeypatch.setenv("RADIO_GS_MPR_LOCAL_MIRROR_DIR", str(mirror))

    cache, _digest, _source = load_mpr_cache(manifest)

    assert isinstance(cache, ShardedMPRCache)
    assert all(shard.path.parent == authority for shard in cache.shards)
    assert all(shard.mapped_path.parent == mirror for shard in cache.shards)
    provenance = cache.provenance()
    assert Path(provenance["support"]["path"]).parent == authority.resolve()
    assert all(Path(record["path"]).parent == authority for record in provenance["shards"])
    assert not any(str(mirror) in str(value) for value in provenance["shards"])
    torch.testing.assert_close(
        cache.fetch_rows(torch.tensor([3, 0])), dense[[3, 0]], rtol=0, atol=0
    )


def test_sharded_cache_rejects_local_mirror_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    manifest = _write_sharded_cache(authority, torch.randn(4, 5))
    mirror = tmp_path / "mirror"
    names = _copy_sharded_members_to_mirror(manifest, mirror)
    shard = mirror / names[1]
    corrupted = bytearray(shard.read_bytes())
    corrupted[0] ^= 1
    shard.write_bytes(corrupted)
    monkeypatch.setenv("RADIO_GS_MPR_LOCAL_MIRROR_DIR", str(mirror))

    with pytest.raises(ValueError, match="local mirror shard SHA-256 differs"):
        load_mpr_cache(manifest)


def test_sharded_cache_rejects_local_mirror_member_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    manifest = _write_sharded_cache(authority, torch.randn(4, 5))
    mirror = tmp_path / "mirror"
    names = _copy_sharded_members_to_mirror(manifest, mirror)
    mirrored_shard = mirror / names[1]
    mirrored_shard.unlink()
    mirrored_shard.symlink_to(authority / names[1])
    monkeypatch.setenv("RADIO_GS_MPR_LOCAL_MIRROR_DIR", str(mirror))

    with pytest.raises(ValueError, match="local mirror shard 0 must not be a symlink"):
        load_mpr_cache(manifest)


@pytest.mark.parametrize("corruption", ["missing", "tampered"])
def test_sharded_cache_rejects_missing_or_tampered_shard(
    tmp_path: Path, corruption: str
) -> None:
    manifest = _write_sharded_cache(tmp_path, torch.randn(4, 5))
    record = json.loads(manifest.read_text(encoding="utf-8"))["shards"][0]
    shard = tmp_path / record["relative_path"]
    if corruption == "missing":
        shard.unlink()
    else:
        data = bytearray(shard.read_bytes())
        data[0] ^= 1
        shard.write_bytes(bytes(data))
    with pytest.raises((FileNotFoundError, ValueError), match="shard|artifact"):
        load_mpr_cache(manifest)


def test_streamed_contribution_mean_matches_dense_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps = [
        torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [0.0, 1.0]],
                [[3.0, 4.0], [5.0, 6.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ]
        ),
        torch.tensor(
            [
                [[0.5, 1.5], [2.5, 3.5]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[4.0, 3.0], [2.0, 1.0]],
                [[2.0, 2.0], [1.0, 1.0]],
            ]
        ),
    ]
    feature_dir = tmp_path / "features"
    (feature_dir / "backbone").mkdir(parents=True)
    records = {}
    for frame, value in zip((3, 7), maps):
        path = feature_dir / "backbone" / f"rgb_{frame}.pt"
        # The production raw descriptor dimension is fixed at 1280.  Repeat
        # the four test channels, then compare only the first four output rows.
        expanded = value.repeat(320, 1, 1)
        torch.save(expanded, path)
        records[f"backbone/rgb_{frame}.pt"] = {"sha256": sha256_file(path)}
    assignments = [
        {
            "gaussian_ids": torch.tensor([0, 1, 2]),
            "pixel_ids": torch.tensor([0, 1, 3]),
            "weights": torch.tensor([1.0, 0.5, 1.0]),
        },
        {
            "gaussian_ids": torch.tensor([0, 1, 2]),
            "pixel_ids": torch.tensor([2, 0, 1]),
            "weights": torch.tensor([0.5, 1.0, 0.25]),
        },
    ]
    output = tmp_path / "raw.pt"
    kwargs = dict(
        output=output,
        feature_space="radio",
        feature_dir=feature_dir,
        feature_tensor_records=records,
        selected_frame_indices=[3, 7],
        feature_size=(2, 2),
        responsibility_assignments=assignments,
        num_gaussians=3,
        output_dim=4,
        shard_channels=2,
        inner_channel_chunk_size=2,
        point_chunk_size=2,
        num_views=2,
        normalize_each_view=False,
        reliability_mode="legacy_valid",
        adaptor=None,
        device=torch.device("cpu"),
        resume_contract={"fixture": "v1"},
    )
    shards, valid, counts, reliability = (
        _stream_channel_sharded_contribution_mean(**kwargs)
    )
    streamed = torch.cat(
        [
            torch.from_numpy(
                np.memmap(
                    tmp_path / str(record["relative_path"]),
                    mode="r",
                    dtype="<f2",
                    shape=tuple(record["shape"]),
                ).copy()
            )
            for record in shards
        ],
        dim=1,
    )
    registered_sum = torch.zeros(3, 4)
    registered_weights = torch.zeros(3)
    for feature_map, assignment in zip(maps, assignments):
        accumulate_contribution_mean_channel_chunked(
            feature_map,
            assignment["gaussian_ids"],
            assignment["pixel_ids"],
            assignment["weights"],
            registered_sum,
            registered_weights,
            channel_chunk_size=2,
        )
    dense, dense_valid = finalize_registered_mean_chunked(
        registered_sum, registered_weights, row_chunk_size=2
    )
    torch.testing.assert_close(streamed, dense, rtol=0, atol=0)
    assert torch.equal(valid, dense_valid)
    assert torch.equal(counts, torch.tensor([2, 2, 2]))
    torch.testing.assert_close(reliability[:, 2], valid.half())

    monkeypatch.setattr(
        "radio_gs.scripts.build_gaussian_multiview_teacher_cache._load_bundle_feature_maps",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed shards must be resumed without feature reads")
        ),
    )
    resumed = _stream_channel_sharded_contribution_mean(**kwargs)
    assert resumed[0] == shards


def _write_responsibility_fixture(path: Path, contract: dict) -> str:
    torch.save(
        {
            "schema_version": 1,
            "metadata": contract,
            "assignments": [
                {
                    "gaussian_ids": torch.tensor([0, 1], dtype=torch.int32),
                    "pixel_ids": torch.tensor([0, 3], dtype=torch.int32),
                    "weights": torch.tensor([1.0, 0.5]),
                }
            ],
        },
        path,
    )
    return sha256_file(path)


def _write_progress_binding(output: Path, responsibility_sha256: str) -> Path:
    resume_contract = {
        "schema": "radio_gs.channel_sharded_mpr_resume.v1",
        "registration_responsibility_cache_sha256": responsibility_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(
            resume_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    progress = output.with_suffix(output.suffix + ".partial.json")
    progress.write_text(
        json.dumps(
            {
                "schema": "radio_gs.channel_sharded_mpr_progress.v1",
                "resume_contract_sha256": digest,
                "resume_contract": resume_contract,
                "shards": [],
            }
        ),
        encoding="utf-8",
    )
    return progress


def test_save_mode_resume_reuses_sha_bound_responsibility(tmp_path: Path) -> None:
    output = tmp_path / "raw.pt"
    sidecar = tmp_path / "responsibility.pt"
    contract = {
        "selected_frame_indices": [3],
        "feature_height": 2,
        "feature_width": 2,
    }
    original_sha256 = _write_responsibility_fixture(sidecar, contract)
    _write_progress_binding(output, original_sha256)

    assignments, observed_sha256, source = (
        _load_saved_responsibility_for_sharded_resume(
            output=output,
            save_path=sidecar,
            expected_contract=contract,
            num_gaussians=2,
        )
    )
    assert observed_sha256 == original_sha256
    assert sha256_file(sidecar) == original_sha256
    assert source == sidecar.resolve()
    assert len(assignments) == 1

    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    payload["assignments"][0]["weights"][0] = 0.75
    torch.save(payload, sidecar)
    with pytest.raises(ValueError, match="SHA-256"):
        _load_saved_responsibility_for_sharded_resume(
            output=output,
            save_path=sidecar,
            expected_contract=contract,
            num_gaussians=2,
        )


def test_save_mode_never_overwrites_an_unbound_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "raw.pt"
    sidecar = tmp_path / "responsibility.pt"
    contract = {
        "selected_frame_indices": [3],
        "feature_height": 2,
        "feature_width": 2,
    }
    original_sha256 = _write_responsibility_fixture(sidecar, contract)
    with pytest.raises(ValueError, match="without a sharded progress binding"):
        _load_saved_responsibility_for_sharded_resume(
            output=output,
            save_path=sidecar,
            expected_contract=contract,
            num_gaussians=2,
        )
    assert sha256_file(sidecar) == original_sha256


def test_stream_initializes_empty_progress_before_first_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "raw.pt"
    monkeypatch.setattr(
        "radio_gs.scripts.build_gaussian_multiview_teacher_cache._load_bundle_feature_maps",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected stop")),
    )
    with pytest.raises(RuntimeError, match="injected stop"):
        _stream_channel_sharded_contribution_mean(
            output=output,
            feature_space="radio",
            feature_dir=tmp_path,
            feature_tensor_records={},
            selected_frame_indices=[3],
            feature_size=(2, 2),
            responsibility_assignments=[
                {
                    "gaussian_ids": torch.tensor([0]),
                    "pixel_ids": torch.tensor([0]),
                    "weights": torch.tensor([1.0]),
                }
            ],
            num_gaussians=1,
            output_dim=2,
            shard_channels=2,
            inner_channel_chunk_size=2,
            point_chunk_size=1,
            num_views=1,
            normalize_each_view=False,
            reliability_mode="legacy_valid",
            adaptor=None,
            device=torch.device("cpu"),
            resume_contract={"fixture": "empty-progress-v1"},
        )
    progress = json.loads(
        output.with_suffix(output.suffix + ".partial.json").read_text(
            encoding="utf-8"
        )
    )
    assert progress["shards"] == []


def test_dense_and_sharded_training_losses_are_equivalent(tmp_path: Path) -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.5, -1.0],
            [0.0, 1.0, -0.5, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 1.0, 1.0],
        ]
    ).half()
    path = _write_sharded_cache(tmp_path, features, feature_space="dino_v3")
    sharded, _digest, _source = load_mpr_cache(
        path, expected_feature_space="dino_v3"
    )
    assert isinstance(sharded, ShardedMPRCache)
    dense = PrimitiveConsensus(
        targets=features,
        valid=sharded.valid,
        observation_count=sharded.observation_count,
        reliability=sharded.reliability,
        per_view_agreement=torch.empty(0, features.shape[0]),
    )
    rows = torch.tensor([3, 0, 1])
    predicted = features[rows].float() + 0.1
    dense_raw, _ = primitive_reconstruction_loss(
        predicted, dense, row_indices=rows
    )
    sharded_raw, _ = primitive_reconstruction_loss(
        predicted, sharded, row_indices=rows
    )
    torch.testing.assert_close(sharded_raw, dense_raw, rtol=0, atol=0)
    dense_capability = _capability_consensus_loss(predicted, dense, rows)[0]
    sharded_capability = _capability_consensus_loss(predicted, sharded, rows)[0]
    torch.testing.assert_close(
        sharded_capability, dense_capability, rtol=0, atol=0
    )


def test_sharded_pca_materializes_the_same_frozen_sample(tmp_path: Path) -> None:
    features = torch.arange(80, dtype=torch.float32).reshape(10, 8).half()
    counts = torch.tensor([2, 1, 2, 1, 2, 0, 2, 1, 2, 1])
    path = _write_sharded_cache(
        tmp_path, features, counts=counts, shard_channels=3
    )
    sharded, _digest, _source = load_mpr_cache(path)
    assert isinstance(sharded, ShardedMPRCache)
    valid_rows = torch.where(counts > 0)[0]
    generator = torch.Generator(device="cpu").manual_seed(17)
    chosen = torch.randperm(valid_rows.numel(), generator=generator)[:4]
    expected = features[valid_rows[chosen]].float()
    actual = _basis_fit_values(
        sharded, valid_rows, max_samples=4, seed=17
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_trainer_streams_raw_target_and_binds_all_shards(tmp_path: Path) -> None:
    features = torch.randn(12, 8).half()
    counts = torch.tensor([2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1])
    cache = _write_sharded_cache(
        tmp_path, features, counts=counts, shard_channels=3
    )
    checkpoint = tmp_path / "radio.ckpt"
    checkpoint.write_bytes(b"test-only checkpoint identity")
    output = tmp_path / "field.pth"
    args = SimpleNamespace(
        mpr_cache=str(cache),
        expected_mpr_cache_sha256=sha256_file(cache),
        observation_contract="unchecked",
        radio_checkpoint=str(checkpoint),
        expected_radio_checkpoint_sha256=sha256_file(checkpoint),
        expected_feature_output_bundle_sha256="",
        output=str(output),
        initial_field_checkpoint="",
        expected_initial_field_checkpoint_sha256="",
        radio_version="test",
        device="cpu",
        coefficient_dim=2,
        local_dim=2,
        spatial_coarse_dim=0,
        hash_levels=2,
        hash_features_per_level=2,
        hash_log2_size=4,
        hash_base_resolution=2,
        hash_max_resolution=4,
        hash_hidden_dim=4,
        fusion_reliability=True,
        hidden_dim=8,
        fusion_residual_blocks=0,
        primitive_fusion=False,
        pca_samples=6,
        no_standardize=False,
        freeze_basis=True,
        official_capability_loss=False,
        dino_mpr_cache="",
        expected_dino_v3_mpr_cache_sha256="",
        sam3_mpr_cache="",
        expected_sam3_mpr_cache_sha256="",
        mpr_weight=1.0,
        dino_weight=0.2,
        sam3_weight=0.2,
        coefficient_weight=0.0,
        basis_orthogonality_weight=0.0,
        epochs=1,
        min_epochs=1,
        batch_size=4,
        eval_batch_size=5,
        learning_rate=1e-3,
        weight_decay=0.0,
        validation_fraction=0.2,
        target_cosine=2.0,
        seed=11,
    )
    report = train(args)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert report["num_gaussians"] == 12
    storage = payload["mpr_cache_storage"]
    assert storage["storage"] == "radio_gs.channel_sharded_mpr.v1"
    assert storage["manifest_sha256"] == sha256_file(cache)
    assert len(storage["shards"]) == 3
    assert all(len(record["sha256"]) == 64 for record in storage["shards"])
