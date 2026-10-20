import json

import pytest
import torch
from torch.nn import functional as F

from radio_gs.query_field.contracts import QueryPacket
from radio_gs.query_field.shared_storage import load_field_v2, save_field_v2
from radio_gs.query_field.unified_modes import UnifiedModesField


def fixture():
    torch.manual_seed(9)
    n = 11
    geometry = dict(xyz=torch.randn(n, 3), normals=torch.randn(n, 3),
                    quaternions=F.normalize(torch.randn(n, 4), dim=-1),
                    scales=torch.ones(n, 3), opacities=torch.ones(n, 1))
    return UnifiedModesField(geometry, torch.arange(n) != 10, torch.randn(n, 4),
                             semantic_dim=7, local_dim=6, hidden_dim=8, mode_count=3, temperature=5.)


def test_menu_order_and_chunking_and_single_latent_gradients():
    field = fixture()
    queries = torch.randn(5, 7)
    p = field.category_probabilities(queries, chunk_size=3)
    perm = torch.tensor([2, 4, 1, 3, 0])
    assert torch.allclose(p[:, perm], field.category_probabilities(queries[perm], chunk_size=7), atol=1e-6)
    assert torch.allclose(p.sum(-1), torch.ones(11), atol=1e-6)
    (-p[:, 0].log().mean()).backward()
    assert field.latent.grad.abs().sum() > 0
    assert all(head.weight.grad.abs().sum() > 0 for head in field.mode_heads)
    assert field.mode_mass.weight.grad.abs().sum() > 0
    assert [k for k, v in field.named_parameters() if v.shape[0] == 11] == ["latent"]
    assert not any("descriptor" in k or "codes" in k for k in field.state_dict())
    assert field.category_partition(queries)[1][-1] == -1


def test_competition_before_integration_represents_mixture_that_mean_cannot():
    field = fixture()
    with torch.no_grad():
        for head in field.mode_heads:
            head.weight.zero_()
            head.bias.zero_()
        field.mode_heads[0].bias[0] = 1
        field.mode_heads[1].bias[1] = 1
        field.mode_heads[2].bias[1] = 1
        field.mode_mass.weight.zero_()
        field.mode_mass.bias.copy_(torch.tensor([.5, .25, .25]).log())
    queries = torch.eye(7)[:3]
    field.temperature = 100.
    field.integration = "reliable"
    mixture = field.category_probabilities(queries)
    assert torch.allclose(mixture[0], torch.tensor([.5, .5, 0.]), atol=1e-6)
    # A third direction between the first two wins after averaging, despite no
    # mode preferring it. This specifically detects the lost nonlinear order.
    queries[2] = F.normalize(queries[0] + queries[1], dim=0)
    mixture = field.category_probabilities(queries)
    field.integration = "mean"
    collapsed = field.category_probabilities(queries)
    assert mixture[0, 2] < .001
    assert collapsed[0, 2] > .999


def test_uniform_menu_and_invalid_inputs():
    field = fixture()
    assert torch.allclose(field.category_probabilities(torch.zeros(4, 7)), torch.full((11, 4), .25))
    for queries in (torch.ones(1, 7), torch.ones(3, 6), torch.full((3, 7), torch.nan)):
        with pytest.raises(ValueError):
            field.category_probabilities(queries)


def test_v2_cold_export_has_no_teacher_dependency(tmp_path, monkeypatch):
    import builtins
    import io
    field = fixture()
    save_field_v2(field, tmp_path / "field", {}, store_root=tmp_path / "blobs", reserve_bytes=1)
    loaded, manifest = load_field_v2(tmp_path / "field", store_root=tmp_path / "blobs")
    queries = torch.randn(4, 7)
    expected = field.category_probabilities(queries)
    packet = QueryPacket("text", semantic=queries[0])
    selected = field.query(packet).selected
    assert manifest["spec"]["architecture"] == "unified_compact_modes_v2"

    def denied(*args, **kwargs):
        raise AssertionError("query attempted filesystem access")

    monkeypatch.setattr(builtins, "open", denied)
    monkeypatch.setattr(io, "open", denied)
    monkeypatch.setattr(torch, "load", denied)
    assert torch.equal(expected, loaded.category_probabilities(queries))
    assert torch.equal(selected, loaded.query(packet).selected)


def test_dictionary_only_stores_shared_basis_and_one_row_latent(tmp_path, monkeypatch):
    import builtins
    import io
    from radio_gs.query_field.contracts import Support
    from radio_gs.query_field.unified_dictionary import UnifiedDictionaryField
    base = fixture()
    field = UnifiedDictionaryField({k: getattr(base, k) for k in
        ("xyz", "normals", "quaternions", "scales", "opacities")}, base.observed, base.latent,
        semantic_dim=7, local_dim=6, hidden_dim=8, semantic_atoms=F.normalize(torch.randn(5, 7), dim=-1))
    assert [k for k, v in field.named_parameters() if v.shape[0] == 11] == ["latent"]
    assert not field.semantic_atoms.requires_grad
    queries = torch.randn(3, 7)
    field.category_probabilities(queries)[:, 0].mean().backward()
    assert field.latent.grad.abs().sum() > 0
    save_field_v2(field, tmp_path / "field", {}, store_root=tmp_path / "blobs", reserve_bytes=1)
    loaded, _ = load_field_v2(tmp_path / "field", store_root=tmp_path / "blobs")
    packets = [QueryPacket("text", semantic=queries[0]),
               QueryPacket("image", semantic=queries[0], local=torch.randn(2, 6)),
               QueryPacket("point_2d", positive=Support(torch.tensor([2, 3]), torch.tensor([.4, .6]))),
               QueryPacket("point_3d", positive=Support(torch.tensor([2]), torch.ones(1)), exact_positive=True)]
    expected = [field.query(q).membership_scores for q in packets]
    scores = field.category_probabilities(queries)

    def denied(*args, **kwargs):
        raise AssertionError("unexpected source access")

    monkeypatch.setattr(builtins, "open", denied)
    monkeypatch.setattr(io, "open", denied)
    monkeypatch.setattr(torch, "load", denied)
    assert torch.equal(scores, loaded.category_probabilities(queries))
    for q, target in zip(packets, expected):
        assert torch.equal(target, loaded.query(q).membership_scores)


@pytest.mark.parametrize("decoder", ["free_modes", "native_dictionary"])
def test_tiny_menu_transfer_pipeline_and_teacher_subset(tmp_path, monkeypatch, decoder):
    from test_unified_training_pipeline import make_source
    from radio_gs.query_field.storage import sha256
    from radio_gs.query_field.bounded_experiment import BoundedExperiment
    from radio_gs.query_field.semantic_measure import SemanticMeasure
    from radio_gs.query_field.unified_training import initialize as initialize_base
    from radio_gs.query_field.unified_menu_training import MeasureTeacher, run

    initial = make_source(tmp_path / "source")
    source = tmp_path / "source"
    policy = dict(study_bytes=64_000_000, run_bytes=40_000_000, artifact_bytes=34_000_000, reserve_bytes=1)
    base = initialize_base(initial, dict(seed=20, latent_dim=4, pca_rows=32, hidden_dim=8), "cpu")
    store = BoundedExperiment(tmp_path / "study", "base", policy)
    exported = store.export(base, {}, max_new_bytes=2_000_000)
    store.json("design.json", dict(source_manifest_sha256=sha256(source / "observations.json")))
    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()
    descriptors = F.normalize(torch.randn(20, 7), dim=-1)
    state = {k: v for k, v in initial["geometry"].items()}
    state.update({"semantic.codes": initial["semantic"],
                  "semantic_measure.descriptors": descriptors,
                  "semantic_measure.pointer": torch.cat([torch.arange(0, 287, 2), torch.tensor([286])]),
                  "semantic_measure.index": torch.arange(286) % 20,
                  "semantic_measure.mass": torch.full((286,), .5),
                  "semantic_measure.supported": torch.arange(144) < 143})
    torch.save(state, teacher_root / "field.pt")
    header = tmp_path / "frame_00001.pt"
    torch.save(dict(frame=1, query_independent=True, descriptors=torch.randn(9, 7)), header)
    manifest = dict(files={"field.pt": dict(sha256=sha256(teacher_root / "field.pt"), bytes=(teacher_root / "field.pt").stat().st_size)},
                    provenance=dict(query_free=True, labels_used=False, legal_frames=[0], excluded_frames=[1],
                                    source_headers=[dict(path=str(header), sha256=sha256(header))]))
    (teacher_root / "manifest.json").write_text(json.dumps(manifest))
    teacher = MeasureTeacher(teacher_root, sha256(teacher_root / "manifest.json"), base, "cpu")
    measure = SemanticMeasure({k.removeprefix("semantic_measure."): v for k, v in state.items()
                              if k.startswith("semantic_measure.")}, 144)
    q = teacher.dev_queries[:4]
    rows = torch.tensor([3, 8, 3, 143])
    fallback = (100 * F.normalize(initial["semantic"], dim=-1) @ q.T).softmax(-1)
    expected = measure.probabilities(q, fallback, "reliable")[rows]
    assert torch.allclose(teacher.probabilities(rows, q), expected, atol=1e-6)
    config = dict(observations=str(source), teacher=str(teacher_root),
                  teacher_manifest_sha256=sha256(teacher_root / "manifest.json"), base_run="base",
                  base_state_identity=exported["state_identity"], arms=["mean", "reliable"], mode_count=3,
                  mode_initial_noise=.01, temperature=100., seed=21, steps=2, learning_rate=.001,
                  training_rows=8, menu_sizes=[3, 5], pairs=2, validation_every=1,
                  validation_menus=2, validation_rows=8, validation_episodes=2,
                  preservation_weight=.25, membership_weight=.25,
                  checkpoint_budget_bytes=1_000_000, export_budget_bytes=2_000_000, storage=policy)
    if decoder == "native_dictionary":
        config.update(decoder=decoder, atom_count=4, basis_initial_rows=32, basis_initial_ridge=.001, mass_weight=1.,
                      arms=["mean", "reliable", "reliable_no_mass"])
        from radio_gs.query_field.unified_dictionary import compress_teacher
        atoms, _ = compress_teacher(teacher, config["atom_count"], config["seed"])
        basis_store = BoundedExperiment(tmp_path / "study", "basis", policy)
        basis_store.tensor("basis.pt", dict(atoms=atoms), max_bytes=5000)
        config["basis_artifact"] = dict(run="basis", file="basis.pt", sha256=sha256(basis_store.root / "basis.pt"))
    reports = {arm: run(config, arm, tmp_path / "study", run_name=arm) for arm in config["arms"]}
    for arm, report in reports.items():
        assert report["single_gaussian_latent"]
        assert not (tmp_path / "study" / arm / "best.pt").exists()
        assert (tmp_path / "study" / arm / "field/manifest.json").exists()
        assert report["selected_validation"]["menu_kl"] >= -1e-6
    # Exercise the real evaluator, replacing only target-label scoring.
    from radio_gs.query_field import public_scannet, unified_menu_evaluation
    queries = [dict(text="first", class_id=1), dict(text="second", class_id=2)]
    query_path, eval_path = tmp_path / "queries.json", tmp_path / "evaluation.json"
    query_path.write_text(json.dumps(dict(queries=queries)))
    eval_path.write_text("{}")
    previous = BoundedExperiment(tmp_path / "study", "encoded", policy)
    previous.tensor("query_vectors.pt", dict(queries=queries, vectors=torch.randn(2, 7)), max_bytes=5000)
    previous.json("query_encoding.json", dict(parameter_bytes=100, query_vectors_sha256=sha256(previous.root / "query_vectors.pt")))
    previous.json("design.json", dict(queries_sha256=sha256(query_path)))
    scored = []

    def scorer(predictions, evaluation, output):
        barrier = json.loads((tmp_path / "study/menu_comparison/prediction_barrier.json").read_text())
        expected = {"mean", "reliable", "base_v1", "teacher_same_domain"}
        if decoder == "native_dictionary":
            expected.update({"basis_oracle", "reliable_no_mass"})
        assert set(barrier) == expected
        for arm, digest in barrier.items():
            assert sha256(tmp_path / "study" / ("menu_comparison_" + arm) / "predictions.pt") == digest
        scored.append(predictions)
        return dict(metrics={"native": {"19": {"miou": .25}}})

    monkeypatch.setattr(public_scannet, "score", scorer)
    summary = unified_menu_evaluation.evaluate(config, tmp_path / "study", {a: a for a in config["arms"]},
                 query_path, eval_path, query_comparison="encoded")
    assert len(scored) == (6 if decoder == "native_dictionary" else 4)
    assert summary["all_predictions_sealed_before_score"]
    assert not summary["five_benchmarks_complete"]
