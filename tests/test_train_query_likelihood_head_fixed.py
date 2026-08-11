from __future__ import annotations

import argparse
import json

import pytest
import torch

from radio_gs.scripts.evaluate_query_likelihood_head_development import run as run_dev
from radio_gs.scripts.train_query_likelihood_head_fixed import (
    RECIPE,
    _binary_metrics,
    _write_torch_no_clobber,
)


def test_fixed_recipe_matches_original_point_smoke_recipe() -> None:
    assert RECIPE == {
        "recipe_id": "monotone-query-likelihood-adam-seed0-e3-lr0.05-v1",
        "seed": 0,
        "optimizer": "Adam",
        "epochs": 3,
        "learning_rate": 0.05,
        "example_order": "sealed_manifest_then_click_ascending_no_shuffle",
        "objective": "unweighted_primitive_binary_cross_entropy",
        "probability_clamp": [1e-6, 1.0 - 1e-6],
    }


def test_binary_metrics_do_not_confuse_accuracy_with_iou() -> None:
    metrics = _binary_metrics(
        torch.tensor([0.9, 0.1, 0.1, 0.1]),
        torch.tensor([True, True, False, False]),
    )
    assert metrics["accuracy_at_0.5"] == pytest.approx(0.75)
    assert metrics["iou_at_0.5"] == pytest.approx(0.5)
    assert metrics["precision_at_0.5"] == pytest.approx(1.0)
    assert metrics["recall_at_0.5"] == pytest.approx(0.5)


def test_checkpoint_writer_is_no_clobber(tmp_path) -> None:
    path = tmp_path / "head.pt"
    _write_torch_no_clobber(path, {"schema_version": 1, "weight": torch.ones(1)})
    with pytest.raises(ValueError, match="refusing to replace"):
        _write_torch_no_clobber(path, {"schema_version": 1})


def test_development_one_shot_rejects_fit_scene_before_opening_shard(tmp_path) -> None:
    manifest = {
        "artifact_type": "agile3d-query-likelihood-training-dataset-v1",
        "scene_count": 1,
        "records": [
            {
                "scene_id": "scene0000_00",
                "partition": "fit",
                "test_labels_opened": False,
            }
        ],
        "safety": {
            "labels_opened": True,
            "label_scope": "official_source_train_scene_only",
            "test_labels_opened": False,
            "full312_evaluation_authorized": False,
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PermissionError, match="scene0003_00"):
        run_dev(argparse.Namespace(dataset_manifest=str(path)))
