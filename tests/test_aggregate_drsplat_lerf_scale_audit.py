import hashlib
import json
from pathlib import Path

import pytest

from radio_gs.scripts import aggregate_drsplat_lerf_scale_audit as aggregate_module
from radio_gs.scripts.aggregate_drsplat_lerf_scale_audit import (
    AuditError,
    EXPECTED_FRAME_COUNTS,
    EXPECTED_RENDER_COUNTS,
    SCENES,
    aggregate_reports,
    write_json_exclusive,
)


@pytest.fixture(autouse=True)
def _fixture_pq_fingerprint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        aggregate_module,
        "EXPECTED_PQ_SHA256",
        hashlib.sha256(b"pq").hexdigest(),
    )


def _write_scene_fixture(root: Path, scene: str, iou: float) -> Path:
    source = root / "compat_data" / scene
    source.mkdir(parents=True)
    rgb_checkpoint = root / "occam" / scene / "chkpnt30000.pth"
    rgb_checkpoint.parent.mkdir(parents=True)
    rgb_checkpoint.write_bytes(b"rgb")
    pq_index = root / "pq_index.faiss"
    pq_index.write_bytes(b"pq")

    model = root / f"{scene}_3_l3paired_topk45_weight_128"
    prediction_root = model / "predictions_mask_0.4" / "renders_silhouette"
    prediction_root.mkdir(parents=True)
    (model / "chkpnt0.pth").write_bytes(b"checkpoint")
    (model / "cfg_args").write_text(
        "Namespace("
        f"source_path={str(source)!r}, model_path={str(model)!r}, "
        "sh_degree=3, images='images', resolution=-1, data_device='cuda', "
        "language_features_name='language_features_dim3', feature_level=3, "
        "eval=True, iterations=0, test_iterations=[0], "
        "save_iterations=[0, 0], checkpoint_iterations=[0], "
        f"start_checkpoint={str(rgb_checkpoint)!r}, name_extra='l3paired', "
        f"mode='mean', topk=45, use_pq=True, pq_index={str(pq_index)!r})",
        encoding="utf-8",
    )

    objects = []
    object_index = 0
    for frame, count in EXPECTED_FRAME_COUNTS[scene].items():
        gt_path = root / "label" / scene / f"{frame}.json"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text("{}", encoding="utf-8")
        for frame_index in range(count):
            query = f"query_{frame_index:02d}"
            pred_path = prediction_root / frame / f"{query}.png"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_bytes(b"png")
            objects.append(
                {
                    "frame": frame,
                    "query": query,
                    "gt_path": str(gt_path),
                    "pred_path": str(pred_path),
                    "iou": iou,
                    "missing": False,
                }
            )
            object_index += 1

    extra_count = EXPECTED_RENDER_COUNTS[scene] - object_index
    first_frame = next(iter(EXPECTED_FRAME_COUNTS[scene]))
    for extra_index in range(extra_count):
        extra_path = prediction_root / first_frame / f"extra_{extra_index:03d}.png"
        extra_path.write_bytes(b"png")

    count = len(objects)
    metrics = {
        "miou": iou,
        "acc025": float(iou > 0.25),
        "acc05": float(iou > 0.5),
        "count": count,
        "missing": 0,
    }
    report = {
        "protocol": "Dr. Splat/VALA LERF nested mask IoU",
        "mask_thresh": "0.4",
        "threshold": 10,
        "ablation_type": "none",
        "prediction_dir": "renders_silhouette",
        "scenes": {
            scene: {
                **metrics,
                "objects": objects,
                "pred_root": str(prediction_root),
            }
        },
        "macro": metrics,
    }
    report_path = root / f"{scene}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_aggregate_reports_separates_scene_macro_and_query_micro(tmp_path: Path):
    values = {
        "figurines": 0.8,
        "teatime": 0.6,
        "ramen": 0.2,
        "waldo_kitchen": 0.0,
    }
    reports = {
        scene: _write_scene_fixture(tmp_path, scene, values[scene])
        for scene in SCENES
    }

    result = aggregate_reports(reports)

    macro = result["scene_equal_macro"]
    micro = result["query_micro"]
    assert macro["metrics_fraction"]["miou"] == pytest.approx(0.4)
    expected_micro = sum(
        values[scene] * sum(EXPECTED_FRAME_COUNTS[scene].values())
        for scene in SCENES
    ) / 208
    assert micro["metrics_fraction"]["miou"] == pytest.approx(expected_micro)
    assert micro["objects"] == 208
    assert result["strict_checkpoint_reproduction"] is False
    assert result["evidence_class"] == "scale_paired_compatibility_reproduction"


def test_aggregate_reports_rejects_wrong_feature_level(tmp_path: Path):
    reports = {
        scene: _write_scene_fixture(tmp_path, scene, 0.5) for scene in SCENES
    }
    cfg = tmp_path / "ramen_3_l3paired_topk45_weight_128" / "cfg_args"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace("feature_level=3", "feature_level=1"),
        encoding="utf-8",
    )

    with pytest.raises(AuditError, match="feature_level"):
        aggregate_reports(reports)


def test_aggregate_reports_rejects_missing_prediction(tmp_path: Path):
    reports = {
        scene: _write_scene_fixture(tmp_path, scene, 0.5) for scene in SCENES
    }
    report_path = reports["figurines"]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["scenes"]["figurines"]["objects"][0]["missing"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AuditError, match="marked missing"):
        aggregate_reports(reports)


def test_write_json_exclusive_never_overwrites(tmp_path: Path):
    output = tmp_path / "summary.json"
    digest = write_json_exclusive(output.resolve(), {"value": 1})
    assert len(digest) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}

    with pytest.raises(AuditError, match="overwrite"):
        write_json_exclusive(output.resolve(), {"value": 2})
