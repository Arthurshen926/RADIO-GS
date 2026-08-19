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


def test_semantic_fidelity_chunking_matches_dense_objective_and_gradient() -> None:
    predicted = torch.randn(1, 7, 9, 11, requires_grad=True)
    teacher = torch.randn(7, 9, 11)
    alpha = torch.rand(9, 11)
    valid = alpha >= 0.2

    absolute, centered, pixels = finetune._semantic_fidelity_losses(
        predicted,
        teacher,
        alpha,
        alpha_threshold=0.2,
        pixel_chunk_size=13,
    )
    predicted_pixels = predicted[0].permute(1, 2, 0)[valid]
    teacher_pixels = teacher.permute(1, 2, 0)[valid]
    expected_absolute = 1.0 - torch.nn.functional.cosine_similarity(
        predicted_pixels, teacher_pixels, dim=-1, eps=1e-8
    ).mean()
    expected_centered = 1.0 - torch.nn.functional.cosine_similarity(
        predicted_pixels - predicted_pixels.mean(0, keepdim=True),
        teacher_pixels - teacher_pixels.mean(0, keepdim=True),
        dim=-1,
        eps=1e-8,
    ).mean()

    torch.testing.assert_close(absolute, expected_absolute)
    torch.testing.assert_close(centered, expected_centered)
    assert pixels == int(valid.sum())
    gradient = torch.autograd.grad(absolute + centered, predicted, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(
        expected_absolute + expected_centered, predicted
    )[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-5, atol=1e-6)


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


def test_chunked_offloaded_adamw_keeps_half_parameter_moments_float32() -> None:
    parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 16).half())
    optimizer = torch.optim.AdamW([parameter], lr=0.002, weight_decay=1e-5)
    before = parameter.detach().clone()
    parameter.grad = torch.linspace(-1.0, 1.0, 16).half()

    finetune._offloaded_adamw_step(optimizer, chunk_elements=4)

    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert parameter.dtype == torch.float16
    assert not torch.equal(parameter, before)


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


def test_staged_feature_branch_backward_matches_joint_backward() -> None:
    torch.manual_seed(7)
    reference = torch.nn.Linear(5, 4, bias=False).double()
    staged = torch.nn.Linear(5, 4, bias=False).double()
    staged.load_state_dict(reference.state_dict())
    inputs = torch.randn(3, 5, dtype=torch.float64)

    reference_feature = reference(inputs)
    reference_base = reference_feature.square().mean()
    reference_branch = torch.sin(reference_feature * 1.7).sum() * 0.03
    (reference_base + reference_branch).backward()

    staged_feature = staged(inputs)
    staged_base = staged_feature.square().mean()
    staged_branch = torch.sin(staged_feature * 1.7).sum() * 0.03
    feature_gradient = finetune._staged_feature_branch_gradient(
        staged_branch, staged_feature
    )
    finetune._backward_base_with_feature_gradient(
        staged_base, staged_feature, feature_gradient
    )

    torch.testing.assert_close(
        staged.weight.grad, reference.weight.grad, atol=1e-12, rtol=1e-12
    )


def test_two_staged_feature_branches_match_joint_backward() -> None:
    torch.manual_seed(11)
    reference = torch.nn.Linear(6, 5, bias=False).double()
    staged = torch.nn.Linear(6, 5, bias=False).double()
    staged.load_state_dict(reference.state_dict())
    inputs = torch.randn(4, 6, dtype=torch.float64)

    reference_feature = reference(inputs)
    reference_base = reference_feature.square().mean()
    reference_capability = torch.sin(reference_feature * 0.7).sum() * 0.04
    reference_semantic = torch.cos(reference_feature * 1.3).sum() * 0.02
    (reference_base + reference_capability + reference_semantic).backward()

    staged_feature = staged(inputs)
    staged_base = staged_feature.square().mean()
    staged_capability = torch.sin(staged_feature * 0.7).sum() * 0.04
    staged_semantic = torch.cos(staged_feature * 1.3).sum() * 0.02
    capability_gradient = finetune._staged_feature_branch_gradient(
        staged_capability, staged_feature
    )
    semantic_gradient = finetune._staged_feature_branch_gradient(
        staged_semantic, staged_feature
    )
    finetune._backward_base_with_feature_gradient(
        staged_base,
        staged_feature,
        capability_gradient + semantic_gradient,
    )

    torch.testing.assert_close(
        staged.weight.grad, reference.weight.grad, atol=1e-12, rtol=1e-12
    )


def test_move_frozen_modules_preserves_values_and_grad_contract() -> None:
    module = torch.nn.Linear(3, 2).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    expected = {name: value.detach().clone() for name, value in module.state_dict().items()}

    finetune._move_frozen_modules({"fixture": module}, torch.device("cpu"))

    assert all(not parameter.requires_grad for parameter in module.parameters())
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_cpu_gradient_offloaded_adamw_matches_native_step() -> None:
    native_parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 12).reshape(3, 4))
    staged_parameter = torch.nn.Parameter(native_parameter.detach().clone())
    gradient = torch.linspace(0.2, -0.1, 12).reshape(3, 4)
    native = torch.optim.AdamW(
        [native_parameter], lr=0.003, weight_decay=0.01, betas=(0.8, 0.95)
    )
    staged = torch.optim.AdamW(
        [staged_parameter], lr=0.003, weight_decay=0.01, betas=(0.8, 0.95)
    )

    native_parameter.grad = gradient.clone()
    native.step()
    finetune._offloaded_adamw_step_cpu_gradients(
        staged,
        {staged_parameter: gradient.clone()},
        chunk_elements=5,
    )

    torch.testing.assert_close(staged_parameter, native_parameter, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        staged.state[staged_parameter]["exp_avg"],
        native.state[native_parameter]["exp_avg"],
    )
    torch.testing.assert_close(
        staged.state[staged_parameter]["exp_avg_sq"],
        native.state[native_parameter]["exp_avg_sq"],
    )


def test_column_staged_direct_field_gradient_matches_joint_backward() -> None:
    class IdentityDecoder(torch.nn.Module):
        def __init__(self, dimension: int) -> None:
            super().__init__()
            self.coefficient_dim = dimension
            self.basis = torch.nn.Parameter(
                torch.eye(dimension), requires_grad=False
            )

        def decode_map(self, value: torch.Tensor) -> torch.Tensor:
            return value

    class DirectField(torch.nn.Module):
        def __init__(self, values: torch.Tensor) -> None:
            super().__init__()
            self.local_codes = torch.nn.Parameter(values.clone())
            self.decoder = IdentityDecoder(values.shape[1])
            self.fusion = None

        def primitive_confidence(self):
            return None

    class LinearRenderer:
        max_channels_per_chunk = 2

        def __init__(self, weights: torch.Tensor) -> None:
            self.weights = weights

        def render_feature_rows(self, _geometry, _viewmat, features, **_kwargs):
            height, width = 2, 3
            rendered = (self.weights @ features).transpose(0, 1).reshape(
                features.shape[1], height, width
            )
            return {
                "feature_map": rendered,
                "alpha_map": torch.ones(height, width),
                "depth_map": torch.zeros(height, width),
            }

    torch.manual_seed(19)
    values = torch.randn(7, 6)
    weights = torch.randn(6, 7)
    selected = torch.tensor([1, 4, 1, 6])
    renderer = LinearRenderer(weights)
    reference = DirectField(values)
    staged = DirectField(values)

    reference_map = renderer.render_feature_rows(
        None, torch.eye(4), reference.local_codes
    )["feature_map"]
    reference_render_loss = torch.sin(reference_map * 0.8).sum() * 0.03
    reference_mpr_loss = reference.local_codes[selected].square().mean() * 0.2
    (reference_render_loss + reference_mpr_loss).backward()

    staged_result = finetune._render_direct_field_detached_by_columns(
        renderer,
        None,
        staged,
        torch.eye(4),
        feature_height=2,
        feature_width=3,
        reliability_splat=False,
    )
    staged_render_loss = torch.sin(staged_result["feature_map"] * 0.8).sum() * 0.03
    (coefficient_gradient,) = torch.autograd.grad(
        staged_render_loss, staged_result["coefficient_map"]
    )
    selected_codes = staged.local_codes[selected].detach().requires_grad_(True)
    staged_mpr_loss = selected_codes.square().mean() * 0.2
    (selected_gradient,) = torch.autograd.grad(staged_mpr_loss, selected_codes)
    staged_gradient, _norm = finetune._column_staged_direct_field_gradient(
        renderer,
        None,
        staged,
        torch.eye(4),
        coefficient_gradient,
        feature_height=2,
        feature_width=3,
        reliability_splat=False,
        selected_rows=selected,
        selected_row_gradient=selected_gradient,
        grad_clip=1e6,
    )

    torch.testing.assert_close(
        staged_result["feature_map"], reference_map.detach(), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        staged_gradient, reference.local_codes.grad, atol=1e-7, rtol=1e-6
    )


def test_staged_feature_branch_rejects_gradient_shape_drift() -> None:
    feature = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="shape differs"):
        finetune._backward_base_with_feature_gradient(
            feature.square().mean(), feature, torch.ones(2, 2)
        )


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
