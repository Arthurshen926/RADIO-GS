import json

import pytest
import torch
from torch.nn import functional as F

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.query_field.contracts import QueryPacket, Support
from radio_gs.query_field.shared_storage import save_field_v2, load_field_v2
from radio_gs.query_field.unified_compact import UnifiedCompactField


def field_fixture():
    torch.manual_seed(4)
    n = 12
    geometry = dict(xyz=torch.randn(n, 3), normals=torch.randn(n, 3),
                    quaternions=F.normalize(torch.randn(n, 4), dim=-1),
                    scales=torch.ones(n, 3), opacities=torch.ones(n, 1))
    return UnifiedCompactField(geometry, torch.arange(n) != 11, torch.randn(n, 4),
                               semantic_dim=7, local_dim=6, hidden_dim=8)


def test_both_teachers_reach_one_latent_without_per_row_side_features():
    field = field_fixture()
    for channel in ("semantic", "local"):
        field.zero_grad()
        getattr(field, channel).decode(torch.tensor([1, 3])).square().mean().backward()
        assert field.latent.grad[[1, 3]].abs().sum() > 0
        assert field.latent.grad[[0, 2]].abs().sum() == 0
    assert not any("codes" in key for key in field.state_dict())
    assert [key for key, value in field.named_parameters() if value.shape[0] == 12] == ["latent"]


def test_chunked_native_response_equals_direct_formula():
    field = field_fixture()
    queries = torch.randn(3, 7)
    expected = F.normalize(F.linear(field.latent, field.heads["semantic"].weight,
                                    field.heads["semantic"].bias), dim=-1) @ F.normalize(queries, dim=-1).T
    assert torch.allclose(field.semantic.score(queries, chunk_size=5), expected, atol=1e-6)


def test_shared_storage_roundtrip_and_geometry_dedup(tmp_path):
    field = field_fixture()
    store = tmp_path / "blobs"
    a = save_field_v2(field, tmp_path / "a", {}, store_root=store, reserve_bytes=1)
    field.latent.data.add_(.03)
    b = save_field_v2(field, tmp_path / "b", {}, store_root=store, reserve_bytes=1)
    assert b["new_blob_bytes"] == field.latent.numel() * field.latent.element_size()
    restored, _ = load_field_v2(tmp_path / "b", store_root=store)
    for key, value in field.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    query = QueryPacket("text", semantic=torch.randn(7))
    assert torch.equal(field.query(query).selected, restored.query(query).selected)
    assert a["row_contract"]["geometry"] == b["row_contract"]["geometry"]


def test_prompt_constraints_and_unknown_output_domain():
    field = field_fixture()
    positive = Support(torch.tensor([2]), torch.ones(1))
    negative = Support(torch.tensor([5]), torch.ones(1))
    result = field.query(QueryPacket("point_3d", positive=positive, negative=negative, exact_positive=True))
    assert result.membership_scores[2] == 1
    assert result.membership_scores[5] == 0
    assert result.membership_scores[11] == 0
    uncertain = field.query(QueryPacket("point_2d", positive=positive))
    assert 0 < uncertain.membership_scores[2] < 1
    assert not uncertain.diagnostics["hard_positive_constraint"]


def test_zero_weight_image_patch_is_ignored():
    field = field_fixture()
    semantic, patches = torch.randn(7), torch.randn(2, 6)
    a = field.query(QueryPacket("image", semantic=semantic, local=patches, local_weights=torch.tensor([1., 0.])))
    b = field.query(QueryPacket("image", semantic=semantic, local=patches[:1]))
    assert torch.equal(a.selected, b.selected)
    assert torch.allclose(a.membership_scores, b.membership_scores)


def policy():
    return dict(study_bytes=10_000_000, run_bytes=5_000_000, artifact_bytes=2_000_000, reserve_bytes=1)


def test_failed_atomic_rotation_keeps_previous_checkpoint(tmp_path):
    store = BoundedExperiment(tmp_path / "study", "run", policy())
    store.tensor("latest.pt", {"x": torch.arange(10)}, max_bytes=5000, replace=True)
    previous = (store.root / "latest.pt").read_bytes()
    with pytest.raises(Exception, match="budget|file write|unexpected pos"):
        store.tensor("latest.pt", {"x": torch.ones(10000)}, max_bytes=5000, replace=True)
    assert (store.root / "latest.pt").read_bytes() == previous
    assert not list(store.root.glob(".pending-*"))


def test_budget_includes_old_and_atomic_replacement(tmp_path):
    store = BoundedExperiment(tmp_path / "study", "run", policy())
    store.tensor("latest.pt", {"x": torch.arange(10)}, max_bytes=5000, replace=True)
    store.policy["run_bytes"] = tree_bytes(store.root) + 4999
    with pytest.raises(ValueError, match="run storage"):
        store.tensor("latest.pt", {}, max_bytes=5000, replace=True)


def test_refuses_historical_root_and_protected_replacement(tmp_path):
    historical = tmp_path / "history"
    historical.mkdir()
    (historical / "old.pt").write_bytes(b"retained")
    with pytest.raises(ValueError, match="unmanaged"):
        BoundedExperiment(historical, "new", policy())
    store = BoundedExperiment(tmp_path / "study", "run", policy())
    with pytest.raises(ValueError, match="rolling"):
        store.json("design.json", {}, replace=True)
    assert (historical / "old.pt").read_bytes() == b"retained"


def test_checkpoint_retirement_requires_exact_export(tmp_path):
    field = field_fixture()
    store = BoundedExperiment(tmp_path / "study", "run", policy())
    store.tensor("best.pt", {"model": field.state_dict()}, max_bytes=50_000)
    store.tensor("latest.pt", {}, max_bytes=5000)
    exported = store.export(field, {}, max_new_bytes=1_500_000)
    field.latent.data.add_(1)
    store.tensor("best.pt", {"model": field.state_dict()}, max_bytes=50_000, replace=True)
    with pytest.raises(ValueError, match="differs"):
        store.retire_checkpoints_after_verified_export(exported["state_identity"])
    assert (store.root / "best.pt").exists()
    assert (store.root / "latest.pt").exists()


def test_shared_study_cap_covers_multiple_runs(tmp_path):
    p = policy()
    a = BoundedExperiment(tmp_path / "study", "a", p)
    a.tensor("result.pt", {"x": torch.ones(1000)}, max_bytes=10_000)
    b = BoundedExperiment(tmp_path / "study", "b", p)
    b.policy["study_bytes"] = tree_bytes(b.study) + 9999
    with pytest.raises(ValueError, match="study storage"):
        b.tensor("result.pt", {}, max_bytes=10_000)


def test_scipy_int32_rows_keep_all_positive_ray_contributors():
    from radio_gs.query_field.unified_training import sampled_support
    op = torch.sparse_coo_tensor(torch.tensor([[0, 0, 1], [0, 2, 1]]),
                                 torch.tensor([.3, .7, 1.]), (2, 3)).coalesce()
    support = sampled_support(op, torch.tensor([3, 5, 9], dtype=torch.int32))
    support.validate(12)
    assert support.indices.tolist() == [3, 9]
    assert torch.allclose(support.weights, torch.tensor([.3, .7]))


def test_existing_capability_field_loader_still_roundtrips(tmp_path):
    from radio_gs.query_field.field import CapabilityField
    original = field_fixture()
    field = CapabilityField({k: getattr(original, k) for k in ("xyz", "normals", "quaternions", "scales", "opacities")},
                            torch.randn(12, 7), torch.randn(12, 6), torch.randn(12, 4), original.observed)
    save_field_v2(field, tmp_path / "legacy", {}, reserve_bytes=1)
    restored, _ = load_field_v2(tmp_path / "legacy")
    assert isinstance(restored, CapabilityField)
    for key, value in field.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
