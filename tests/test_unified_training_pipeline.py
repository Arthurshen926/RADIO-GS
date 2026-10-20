"""Exercise actual sparse-source fit/export/evaluation boundaries on a tiny scene."""
import json

import torch
from torch.nn import functional as F

from radio_gs.query_field.contracts import QueryPacket
from radio_gs.query_field.storage import sha256
from radio_gs.query_field.unified_training import run


def make_source(root):
    root.mkdir()
    torch.manual_seed(19)
    n = 144
    initial = dict(geometry=dict(xyz=torch.randn(n, 3), normals=torch.randn(n, 3),
                    quaternions=F.normalize(torch.randn(n, 4), dim=-1),
                    scales=torch.ones(n, 3), opacities=torch.ones(n, 1)),
                   semantic=torch.randn(n, 7), local=torch.randn(n, 6), observed=torch.ones(n, dtype=torch.bool))
    torch.save(initial, root / "initial.pt")
    masks = torch.zeros(2, 12, 12, dtype=torch.bool)
    masks[0, :, :5] = True
    masks[1, :, 7:] = True
    frames = []
    for frame, split in [(0, "train"), (1, "development")]:
        data = dict(operator=torch.eye(n).to_sparse(), raw_mass=torch.ones(n),
                    best_element=torch.arange(n), masks=masks, scales_m=torch.ones(2),
                    semantic_descriptors=torch.randn(2, 7),
                    semantic_pixels=initial["semantic"], local_pixels=initial["local"])
        path = root / f"view_{frame:05d}.pt"
        torch.save(data, path)
        frames.append(dict(frame=frame, split=split, compiled=dict(sha256=sha256(path))))
    manifest = dict(scene="fixture", benchmark_queries_used=False, semantic_labels_used=False,
                    feature_development_excluded=True, world_units_per_metre=1.,
                    splits=dict(train_frames=[0], development_frames=[1]), frames=frames,
                    native_semantic_space="fixture_encoder",
                    initial=dict(sha256=sha256(root / "initial.pt")))
    (root / "observations.json").write_text(json.dumps(manifest))
    return initial


def test_two_arm_pipeline_seals_predictions_before_scoring(tmp_path, monkeypatch):
    from radio_gs.query_field import unified_evaluation, public_scannet
    make_source(tmp_path / "source")
    config = dict(observations=str(tmp_path / "source"), arms=["reconstruction", "functional"],
                  latent_dim=4, hidden_dim=8, seed=20, steps=2, learning_rate=.001,
                  pairs=2, reconstruction_rows=8, pca_rows=32, validation_every=1,
                  validation_episodes=2, functional_weight=1., membership_weight=.25,
                  audit_teacher_gradients=True,
                  checkpoint_budget_bytes=1_000_000, export_budget_bytes=2_000_000,
                  storage=dict(study_bytes=64_000_000, run_bytes=40_000_000,
                               artifact_bytes=34_000_000, reserve_bytes=1))
    study = tmp_path / "study"
    reports = {arm: run(config, arm, study) for arm in config["arms"]}
    assert reports["functional"]["initial_validation"] == reports["reconstruction"]["initial_validation"]
    for arm in config["arms"]:
        folder = study / ("fixture_" + arm)
        assert (folder / "field/manifest.json").exists()
        assert not (folder / "best.pt").exists()
        assert not (folder / "latest.pt").exists()
        audit = json.loads((folder / "teacher_gradient_audit.json").read_text())
        assert set(audit["teachers"]) == {"semantic", "local", "membership"}
        assert all(v["latent_gradient_norm"] > 0 for v in audit["teachers"].values())

    class Encoder:
        def __init__(self, *args):
            self._text = (None, torch.nn.Linear(1, 1))

        def text(self, expression, task="union"):
            return QueryPacket("text", task=task, semantic=torch.arange(1, 8).float() * (1 if expression == "a" else -1))

    calls = []

    def scorer(predictions, evaluation, output):
        barrier = json.loads((study / "comparison/prediction_barrier.json").read_text())
        assert set(barrier) == {"functional", "reconstruction", "native_mean_reference"}
        calls.append(str(predictions))
        return dict(metrics={"native": {"19": {"miou": .25}}})

    monkeypatch.setattr(unified_evaluation, "NativeQueryEncoder", Encoder)
    monkeypatch.setattr(public_scannet, "score", scorer)
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(dict(queries=[dict(text="a", class_id=1), dict(text="b", class_id=2)])))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text("{}")
    result = unified_evaluation.evaluate(config, study, {a: "fixture_" + a for a in config["arms"]}, queries, evaluation)
    assert len(calls) == 3
    assert not result["five_benchmarks_complete"]
