import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import radio_gs.scripts.finetune_canonical_radio_rendering as finetune


def test_selection_snapshot_is_independent_cpu_state() -> None:
    module = torch.nn.Linear(3, 2)
    snapshot = finetune._cpu_state_dict(module)
    expected = {name: value.detach().clone() for name, value in module.state_dict().items()}

    with torch.no_grad():
        module.weight.add_(10.0)

    assert all(value.device.type == "cpu" for value in snapshot.values())
    assert all(torch.equal(snapshot[name], expected[name]) for name in snapshot)
    assert snapshot["weight"].data_ptr() != module.weight.data_ptr()


def test_method_v1_lineage_recovers_base_and_parent_stage(tmp_path: Path) -> None:
    base = tmp_path / "base.pth"
    parent = tmp_path / "spatial.pth"
    base.write_bytes(b"base")
    parent.write_bytes(b"spatial")
    parent_payload = {
        "architecture": {"coefficient_dim": 512, "local_dim": 512},
        "training_config": {"output": str(base)},
        "render_optimization": {
            "selection_policy": "capability_pareto",
            "best_step": 64,
            "official_render_capability": {"adaptor_weights": {"siglip2-g": 0.05}},
            "semantic_capability": {"enabled": False},
            "generic_text_response": {"enabled": False},
        },
    }
    args = SimpleNamespace(field_checkpoint=str(parent), construction_prior_field=[])

    lineage = finetune._method_v1_predecessor_lineage(
        args,
        parent_payload,
        current_stage="genuine_source_crop_region_summary",
    )

    assert [record["stage"] for record in lineage] == [
        "factorized_d512_l512",
        "official_siglip2_full_grid",
    ]
    assert all(len(record["sha256"]) == 64 for record in lineage)


def test_method_v1_lineage_accepts_precomputed_parent_digest(
    tmp_path: Path, monkeypatch
) -> None:
    parent = tmp_path / "base.pth"
    parent.write_bytes(b"base")
    args = SimpleNamespace(field_checkpoint=str(parent), construction_prior_field=[])
    digest = "a" * 64
    monkeypatch.setattr(
        finetune,
        "sha256_file",
        lambda _path: pytest.fail("precomputed digest must avoid a second hash"),
    )

    lineage = finetune._method_v1_predecessor_lineage(
        args,
        {
            "method_v1_construction_lineage": [
                {
                    "stage": "factorized_d512_l512",
                    "field": "/frozen/base.pth",
                    "sha256": "b" * 64,
                    "selection_policy": "mapping_only_checkpoint_rule",
                    "best_step": 0,
                }
            ],
            "render_optimization": {
                "selection_policy": "capability_pareto",
                "best_step": 64,
                "official_render_capability": {
                    "adaptor_weights": {"siglip2-g": 0.05}
                },
                "semantic_capability": {"enabled": False},
                "generic_text_response": {"enabled": False},
            },
        },
        current_stage="genuine_source_crop_region_summary",
        parent_sha256=digest,
    )

    assert lineage[-1]["sha256"] == digest


def test_checkpoint_persists_capability_pareto_drop_authority() -> None:
    source = Path(finetune.__file__).read_text(encoding="utf-8")
    assert '"max_capability_drop": float(args.max_capability_drop)' in source


def test_render_finetune_exposes_official_siglip_spatial_capability() -> None:
    source = Path(finetune.__file__).read_text(encoding="utf-8")
    assert '"siglip2-g": float(args.siglip_spatial_render_weight)' in source
    assert "SigLIP2FeatureProjection.from_radio_checkpoint" in source
    assert "--siglip-spatial-render-weight" in source


def test_mpr_exclusions_accept_nested_registration_authority() -> None:
    metadata = {
        "registration_responsibility_contract": {
            "excluded_frame_ids": [41, 105, 152, 195]
        }
    }

    assert finetune._excluded_mpr_frame_ids(metadata) == {41, 105, 152, 195}


def test_mpr_exclusions_fail_closed_when_authority_is_missing() -> None:
    with pytest.raises(ValueError, match="does not declare"):
        finetune._excluded_mpr_frame_ids({})


def test_training_frame_ids_fall_back_to_frozen_config_authority(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "train_frames.txt"
    authority.write_text("0\n20 # registered\n40\n", encoding="utf-8")
    config = SimpleNamespace(train_frame_ids_path=str(authority))

    assert finetune._resolve_training_frame_ids(config, "") == {0, 20, 40}
    assert finetune._resolve_training_frame_ids(config, "7,9") == {7, 9}


def test_training_frame_ids_accept_frozen_json_authority(tmp_path: Path) -> None:
    authority = tmp_path / "train_frame_ids.json"
    authority.write_text(
        json.dumps({"frame_ids": [0, 2, 4, 7]}), encoding="utf-8"
    )
    config = SimpleNamespace(train_frame_ids_path=str(authority))

    assert finetune._resolve_training_frame_ids(config, "") == {0, 2, 4, 7}


def test_training_frame_ids_fail_closed_for_missing_config_authority(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(train_frame_ids_path=str(tmp_path / "missing.txt"))

    with pytest.raises(FileNotFoundError, match="training frame authority"):
        finetune._resolve_training_frame_ids(config, "")

    empty = tmp_path / "empty.txt"
    empty.write_text("# no registered frames\n", encoding="utf-8")
    config.train_frame_ids_path = str(empty)
    with pytest.raises(ValueError, match="training frame authority is empty"):
        finetune._resolve_training_frame_ids(config, "")


def test_load_consensus_accepts_factorized_radio_cache(tmp_path: Path) -> None:
    path = tmp_path / "factorized.pt"
    torch.save(
        {
            "factorized_radio": {
                "canonical_feature": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "valid": torch.tensor([True, False]),
                "reliability": torch.ones(2, 5),
            },
            "view_counts": torch.tensor([3, 0]),
            "metadata": {"selected_frame_indices": [1]},
        },
        path,
    )

    consensus, payload = finetune._load_consensus(str(path))

    assert consensus.targets.shape == (2, 2)
    assert consensus.reliability.shape == (2, 5)
    assert consensus.observation_count.tolist() == [3, 0]
    assert payload["metadata"]["selected_frame_indices"] == [1]


def test_semantic_fidelity_aligns_one_row_teacher_rounding_difference() -> None:
    """Crop-summary teachers may round the native RADIO height up by one row."""

    predicted = torch.randn(1, 8, 45, 62)
    teacher = torch.randn(8, 46, 62)
    alpha = torch.ones(45, 62)

    absolute, centered, pixels = finetune._semantic_fidelity_losses(
        predicted,
        teacher,
        alpha,
        alpha_threshold=0.02,
    )

    assert pixels == 45 * 62
    assert torch.isfinite(absolute)
    assert torch.isfinite(centered)


def test_semantic_fidelity_rejects_non_rounding_grid_mismatch() -> None:
    predicted = torch.randn(1, 8, 45, 62)
    teacher = torch.randn(8, 47, 62)

    with pytest.raises(ValueError, match="semantic teacher/prediction mismatch"):
        finetune._semantic_fidelity_losses(
            predicted,
            teacher,
            torch.ones(45, 62),
            alpha_threshold=0.02,
        )


class _IdentityAdaptor(torch.nn.Module):
    def forward(self, values):
        return values


class _OneFrameDataset:
    def __init__(self, features: torch.Tensor) -> None:
        self.frame_indices = [7]
        self._features = features

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        assert index == 0
        return {
            "radio_features": self._features.clone(),
            "pose_w2c": torch.eye(4),
        }


def test_capability_validation_uses_direct_official_maps_without_reprojection(
    monkeypatch,
) -> None:
    """The high-fidelity branch compares predictions to official maps directly."""

    predicted = torch.tensor([[[1.0]], [[0.0]]])
    # A deliberately different raw feature proves that direct maps are not
    # sent through the raw adaptor path again.
    raw_dataset = _OneFrameDataset(torch.tensor([[[0.0]], [[1.0]]]))
    official_dataset = _OneFrameDataset(predicted)

    def _render(*_args, **_kwargs):
        return {
            "feature_map": predicted.clone(),
            "alpha_map": torch.ones(1, 1),
        }

    monkeypatch.setattr(finetune, "render_canonical_radio", _render)
    kwargs = {
        "adaptors": {"sam3": _IdentityAdaptor()},
        "reliability_splat": False,
        "alpha_threshold": 0.02,
    }
    legacy = finetune._mean_multicapability_fidelity(
        torch.nn.Identity(),
        None,
        None,
        raw_dataset,
        {7: 0},
        [7],
        torch.device("cpu"),
        **kwargs,
    )
    direct = finetune._mean_multicapability_fidelity(
        torch.nn.Identity(),
        None,
        None,
        raw_dataset,
        {7: 0},
        [7],
        torch.device("cpu"),
        capability_teacher_datasets={"sam3": official_dataset},
        capability_teacher_frame_to_index={"sam3": {7: 0}},
        **kwargs,
    )

    assert legacy["sam3"] == pytest.approx(0.0, abs=1e-6)
    assert direct["sam3"] == pytest.approx(1.0, abs=1e-6)


def test_validation_cuda_cache_release_is_explicit_and_cuda_only(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(True))

    finetune._release_validation_cuda_cache(
        torch.device("cpu"), enabled=True, should_validate=True
    )
    finetune._release_validation_cuda_cache(
        torch.device("cuda:0"), enabled=False, should_validate=True
    )
    finetune._release_validation_cuda_cache(
        torch.device("cuda:0"), enabled=True, should_validate=False
    )
    finetune._release_validation_cuda_cache(
        torch.device("cuda:0"), enabled=True, should_validate=True
    )

    assert calls == [True]


def test_optimizer_state_move_covers_all_tensor_entries() -> None:
    class _Optimizer:
        state = {
            "parameter": {
                "step": torch.tensor(2.0),
                "exp_avg": torch.ones(3),
                "exp_avg_sq": torch.ones(3) * 2,
                "metadata": "retained",
            }
        }

    optimizer = _Optimizer()
    finetune._move_optimizer_state(optimizer, torch.device("meta"))

    assert optimizer.state["parameter"]["step"].device.type == "meta"
    assert optimizer.state["parameter"]["exp_avg"].device.type == "meta"
    assert optimizer.state["parameter"]["exp_avg_sq"].device.type == "meta"
    assert optimizer.state["parameter"]["metadata"] == "retained"


def test_chunked_offloaded_adamw_matches_single_tensor_update() -> None:
    initial = torch.linspace(-0.8, 0.9, 15, dtype=torch.float64).reshape(5, 3)
    reference_parameter = torch.nn.Parameter(initial.clone())
    chunked_parameter = torch.nn.Parameter(initial.clone())
    kwargs = {
        "lr": 0.002,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-5,
        "foreach": False,
    }
    reference = torch.optim.AdamW([reference_parameter], **kwargs)
    chunked = torch.optim.AdamW([chunked_parameter], **kwargs)

    for scale in (0.3, -0.2, 0.7):
        gradient = torch.linspace(-1.0, 1.0, 15, dtype=torch.float64).reshape(5, 3)
        gradient = gradient * scale
        reference_parameter.grad = gradient.clone()
        chunked_parameter.grad = gradient.clone()
        reference.step()
        finetune._offloaded_adamw_step(chunked, chunk_elements=4)

    assert torch.equal(chunked_parameter, reference_parameter)
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(
            chunked.state[chunked_parameter][name],
            reference.state[reference_parameter][name],
        )
        assert chunked.state[chunked_parameter][name].device.type == "cpu"


def test_offloaded_optimizer_reuses_gradient_buffer() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.arange(8, dtype=torch.float32)
    pointer = parameter.grad.data_ptr()

    finetune._zero_optimizer_gradients(optimizer, preserve_buffers=True)

    assert parameter.grad is not None
    assert parameter.grad.data_ptr() == pointer
    assert torch.count_nonzero(parameter.grad) == 0

    finetune._zero_optimizer_gradients(optimizer, preserve_buffers=False)
    assert parameter.grad is None


def test_cpu_selection_snapshot_refreshes_in_place() -> None:
    module = torch.nn.Linear(3, 2)
    snapshot = finetune._cpu_state_dict(module)
    storage_ids = {name: id(value) for name, value in snapshot.items()}
    with torch.no_grad():
        module.weight.fill_(4.0)
        module.bias.fill_(-2.0)

    finetune._copy_state_dict_to_cpu_(snapshot, module)

    assert {name: id(value) for name, value in snapshot.items()} == storage_ids
    assert torch.equal(snapshot["weight"], module.weight.detach().cpu())
    assert torch.equal(snapshot["bias"], module.bias.detach().cpu())


def test_parent_payload_compaction_drops_only_reconstructed_tensors() -> None:
    payload = {
        "state_dict": {"large": torch.ones(3)},
        "reliability": torch.empty(3, 0),
        "architecture": {"coefficient_dim": 512},
        "render_optimization": {"best_step": 32},
    }

    compact = finetune._metadata_only_parent_payload(payload)

    assert "state_dict" not in compact
    assert "reliability" not in compact
    assert compact["architecture"] == payload["architecture"]
    assert compact["render_optimization"] == payload["render_optimization"]


def test_semantic_stage_automatically_enables_optimizer_state_offload() -> None:
    assert finetune._optimizer_state_offload_enabled(
        requested=False, semantic_enabled=True
    )
    assert finetune._optimizer_state_offload_enabled(
        requested=True, semantic_enabled=False
    )
    assert not finetune._optimizer_state_offload_enabled(
        requested=False, semantic_enabled=False
    )
