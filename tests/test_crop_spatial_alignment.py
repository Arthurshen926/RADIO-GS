from argparse import Namespace

import torch
import torch.nn.functional as F

from radio_gs.interfaces.crop_spatial_alignment import (
    GlobalCropContextAdapter,
    GlobalCropSpatialAdapter,
)
from radio_gs.scripts.train_crop_context_alignment import (
    _symmetric_contrastive_loss,
    train as train_context,
)
from radio_gs.scripts.train_crop_spatial_alignment import train
from radio_gs.scripts.build_crop_spatial_alignment_cache import _excluded_scenes


def _cache(path, *, role: str, scene: str) -> None:
    torch.manual_seed(7 if role == "train" else 9)
    crop = F.normalize(torch.randn(8, 6), dim=-1)
    target = F.normalize(crop + 0.05 * torch.randn_like(crop), dim=-1)
    torch.save(
        {
            "schema_version": 1,
            "crop_descriptors": crop.half(),
            "crop_context_descriptors": F.normalize(
                crop + 0.03 * torch.randn_like(crop), dim=-1
            ).half(),
            "full_image_anchor_descriptors": target.half(),
            "metadata": {
                "training_scope": "global_cross_scene_crop_to_spatial_dino",
                "split_role": role,
                "uses_benchmark_scenes": False,
                "uses_benchmark_labels": False,
                "uses_depth": False,
                "uses_pose": False,
                "uses_instances": False,
                "uses_text": False,
                "physical_space_disjoint": True,
                "benchmark_exclusion_declared": True,
                "excluded_physical_spaces": ["scene_test"],
                "scene_names": [scene],
            },
        },
        path,
    )


def test_global_crop_adapter_training_and_fail_closed_reload(tmp_path) -> None:
    train_cache, validation_cache = tmp_path / "train.pt", tmp_path / "val.pt"
    _cache(train_cache, role="train", scene="scene0001_00")
    _cache(validation_cache, role="validation", scene="scene0002_00")
    checkpoint = tmp_path / "adapter.pt"
    report = train(
        Namespace(
            train_caches=[str(train_cache)],
            validation_caches=[str(validation_cache)],
            output=str(checkpoint),
            hidden_dim=2,
            epochs=2,
            patience=0,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=0,
            device="cpu",
        )
    )
    adapter, manifest = GlobalCropSpatialAdapter.from_checkpoint(checkpoint)
    assert report["checkpoint_sha256"] == manifest.checkpoint_sha256
    assert adapter.feature_dim == 6
    assert manifest.scene_disjoint


def test_global_crop_context_adapter_training_and_identity_start(tmp_path) -> None:
    train_cache, validation_cache = tmp_path / "train.pt", tmp_path / "val.pt"
    _cache(train_cache, role="train", scene="scene0001_00")
    _cache(validation_cache, role="validation", scene="scene0002_00")
    checkpoint = tmp_path / "context.pt"
    report = train_context(
        Namespace(
            train_caches=[str(train_cache)],
            validation_caches=[str(validation_cache)],
            output=str(checkpoint),
            hidden_dim=2,
            epochs=2,
            patience=0,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=0,
            device="cpu",
        )
    )
    adapter, manifest = GlobalCropContextAdapter.from_checkpoint(checkpoint)
    probe = F.normalize(torch.randn(3, 6), dim=-1)
    with torch.no_grad():
        untouched = GlobalCropContextAdapter(feature_dim=6, hidden_dim=2)(probe, probe)
    assert torch.allclose(untouched, probe, atol=1e-6)
    assert report["checkpoint_sha256"] == manifest.checkpoint_sha256
    assert adapter.feature_dim == 6
    assert manifest.scene_disjoint


def test_context_contrastive_loss_prefers_aligned_crop_target_pairs() -> None:
    target = torch.eye(3)
    aligned = _symmetric_contrastive_loss(target, target, temperature=0.07)
    swapped = _symmetric_contrastive_loss(
        target, target[[1, 2, 0]], temperature=0.07
    )
    assert aligned < swapped


def test_crop_adapter_exclusion_manifest_is_label_free_and_expands_physical_spaces(tmp_path) -> None:
    manifest = tmp_path / "exclude.json"
    manifest.write_text(
        '{"schema_version":1,"purpose":"global_crop_spatial_adapter_scene_exclusion",'
        '"scene_names":["scene0001_00","scene0001_01"],'
        '"uses_labels":false,"uses_masks":false,"uses_clicks":false,"uses_metrics":false}',
        encoding="utf-8",
    )
    names, record = _excluded_scenes(raw_names="scene0002_00", manifest_path=str(manifest))
    assert names == {"scene0001_00", "scene0001_01", "scene0002_00"}
    assert record is not None and record["scene_count"] == 2
