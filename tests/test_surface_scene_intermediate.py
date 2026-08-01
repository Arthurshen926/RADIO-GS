from dataclasses import replace
import os

import pytest
import torch

import radio_gs.interfaces.surface_scene_intermediate as scene_module
from radio_gs.interfaces.surface_region_contract import SurfaceRegionContractV2
from radio_gs.interfaces.surface_scene_intermediate import (
    EXPECTED_ADAPTOR_ROLES,
    EXPECTED_EDGE_CHANNELS,
    EXPECTED_IMPLEMENTATION_ROLES,
    GEOMETRIC_RELIABILITY_ALGORITHM,
    GEOMETRIC_RELIABILITY_MODE,
    SourceFileBinding,
    SurfaceSceneFrameBinding,
    SurfaceSceneIntermediate,
    SurfaceSceneIntermediateContract,
    assert_exact_surface_scene_replay,
    default_graph_config_dict,
    load_surface_scene_intermediate,
    save_surface_scene_intermediate,
    scientific_tensor_bundle_sha256,
    scientific_tensor_sha256,
    sha256_file,
    tensor_sha256,
)
from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportGraphConfig,
    build_primitive_support_graph,
)


def _write_source(path, payload: str) -> SourceFileBinding:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return SourceFileBinding.from_path(path)


def _contract(tmp_path) -> SurfaceSceneIntermediateContract:
    frames = []
    for frame_index in range(2):
        frame_root = tmp_path / "sources" / f"frame-{frame_index}"
        frames.append(
            SurfaceSceneFrameBinding(
                frame_id=f"{frame_index:06d}",
                color=_write_source(
                    frame_root.with_suffix(".color.jpg"),
                    f"color-{frame_index}",
                ),
                depth=_write_source(
                    frame_root.with_suffix(".depth.png"),
                    f"depth-{frame_index}",
                ),
                pose=_write_source(
                    frame_root.with_suffix(".pose.txt"),
                    f"pose-{frame_index}",
                ),
            )
        )
    implementations = {
        role: _write_source(
            tmp_path / "implementations" / f"{role}.py",
            f"implementation-{role}",
        )
        for role in EXPECTED_IMPLEMENTATION_ROLES
    }
    return SurfaceSceneIntermediateContract(
        scene="scene0024_00",
        source_frames=tuple(frames),
        depth_intrinsic=_write_source(
            tmp_path / "sources" / "intrinsics_depth.txt",
            "depth-intrinsic",
        ),
        color_intrinsic=_write_source(
            tmp_path / "sources" / "intrinsics_color.txt",
            "color-intrinsic",
        ),
        radio_checkpoint=_write_source(
            tmp_path / "checkpoint" / "c-radio.pt",
            "radio-checkpoint",
        ),
        radio_version="c-radio_v4-h",
        radio_resolution=384,
        depth_stride=8,
        voxel_size=0.04,
        adaptor_names={
            "appearance": "dino_v3_7b",
            "boundary": "sam3",
        },
        adaptor_batch_size=64,
        affinity_dimension=256,
        graph_config=default_graph_config_dict(),
        implementation_sources=implementations,
    )


def _graph(
    *,
    edge_index=None,
    raw_affinity=None,
    channels=None,
    edge_weight=None,
    num_nodes=3,
    local_sigma=None,
) -> PrimitiveSupportGraph:
    edges = (
        torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.int64,
        )
        if edge_index is None
        else edge_index
    )
    edge_count = int(edges.shape[1])
    raw = (
        torch.tensor([0.72, 0.72, 0.42, 0.42], dtype=torch.float32)
        if raw_affinity is None
        else raw_affinity
    )
    channel_values = (
        {
            "geometry": torch.tensor(
                [0.9, 0.9, 0.7, 0.7], dtype=torch.float32
            ),
            "appearance": torch.tensor(
                [0.8, 0.8, 0.6, 0.6], dtype=torch.float32
            ),
            "boundary": torch.ones(4, dtype=torch.float32),
        }
        if channels is None
        else channels
    )
    if edge_weight is None:
        row_sum = torch.zeros(num_nodes, dtype=torch.float32)
        row_sum.index_add_(0, edges[0], raw)
        weights = raw / row_sum[edges[0]].clamp_min(1e-12)
    else:
        weights = edge_weight
    sigma = (
        torch.full((num_nodes,), 0.1, dtype=torch.float32)
        if local_sigma is None
        else local_sigma
    )
    assert raw.shape == (edge_count,)
    return PrimitiveSupportGraph(
        edge_index=edges,
        edge_weight=weights,
        raw_affinity=raw,
        local_sigma=sigma,
        num_nodes=num_nodes,
        edge_channels=channel_values,
    )


def _intermediate(tmp_path) -> SurfaceSceneIntermediate:
    features = torch.zeros(3, 1280, dtype=torch.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    features[2, 2] = 1.0
    return SurfaceSceneIntermediate(
        contract=_contract(tmp_path),
        xyz=torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        radio_features=features,
        geometric_reliability=torch.tensor(
            [0.5, 0.75, 1.0], dtype=torch.float32
        ),
        graph=_graph(),
    )


def _torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _trusted_load(path, value, artifact, **kwargs):
    return load_surface_scene_intermediate(
        path,
        expected_contract=value.contract,
        expected_file_sha256=artifact.file_sha256,
        **kwargs,
    )


def test_atomic_round_trip_is_content_addressed_and_exact(tmp_path) -> None:
    fresh = _intermediate(tmp_path)
    output = tmp_path / "cache" / "scene0024_00.pt"

    artifact = save_surface_scene_intermediate(fresh, output)
    replay = _trusted_load(output, fresh, artifact)

    assert artifact.path == str(output.absolute())
    assert artifact.file_sha256 == sha256_file(output)
    assert artifact.contract_sha256 == fresh.contract.digest
    assert (
        artifact.tensor_bundle_sha256
        == scientific_tensor_bundle_sha256(fresh)
    )
    assert dict(artifact.tensor_sha256) == scientific_tensor_sha256(fresh)
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    assert_exact_surface_scene_replay(fresh, replay)
    assert set(replay.graph.edge_channels) == EXPECTED_EDGE_CHANNELS


def test_trusted_load_requires_both_external_authorities(tmp_path) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    artifact = save_surface_scene_intermediate(value, output)

    with pytest.raises(TypeError, match="expected_contract"):
        load_surface_scene_intermediate(output)
    with pytest.raises(TypeError, match="expected_file_sha256"):
        load_surface_scene_intermediate(
            output,
            expected_contract=value.contract,
        )
    with pytest.raises(TypeError, match="expected_contract"):
        load_surface_scene_intermediate(
            output,
            expected_file_sha256=artifact.file_sha256,
        )


def test_contract_binds_order_sources_runtime_roles_and_semantics(tmp_path) -> None:
    contract = _intermediate(tmp_path).contract
    payload = contract.to_dict()

    assert [
        frame["frame_id"] for frame in payload["source_frames"]
    ] == ["000000", "000001"]
    assert set(payload["adaptors"]["role_to_name"]) == EXPECTED_ADAPTOR_ROLES
    assert payload["adaptors"]["affinity_dimension"] == 256
    assert set(payload["implementation_sources"]) == (
        EXPECTED_IMPLEMENTATION_ROLES
    )
    assert payload["radio"]["checkpoint"] == (
        contract.radio_checkpoint.to_dict()
    )
    assert payload["geometric_reliability"] == {
        "mode": GEOMETRIC_RELIABILITY_MODE,
        "algorithm": GEOMETRIC_RELIABILITY_ALGORITHM,
    }
    reversed_contract = replace(
        contract,
        source_frames=tuple(reversed(contract.source_frames)),
    )
    assert reversed_contract.digest != contract.digest


def test_contract_rejects_wrong_authority_sets_and_semantics(tmp_path) -> None:
    contract = _contract(tmp_path)
    missing_adaptor = dict(contract.adaptor_names)
    missing_adaptor.pop("boundary")
    with pytest.raises(ValueError, match="adaptor role mapping keys differ"):
        replace(contract, adaptor_names=missing_adaptor)

    missing_implementation = dict(contract.implementation_sources)
    missing_implementation.pop("support_graph")
    with pytest.raises(ValueError, match="implementation source bindings keys"):
        replace(contract, implementation_sources=missing_implementation)

    with pytest.raises(ValueError, match="reliability mode differs"):
        replace(contract, geometric_reliability_mode="uniform_valid")
    with pytest.raises(ValueError, match="reliability algorithm differs"):
        replace(contract, geometric_reliability_algorithm="other")


@pytest.mark.parametrize(
    ("field", "replacement_value", "message"),
    [
        (
            "topology_mode",
            "mutual_knn",
            "topology_mode must be symmetric_union",
        ),
        (
            "surface_topology_min_affinity",
            0.1,
            "surface_topology_min_affinity must be zero",
        ),
        (
            "require_covisibility_topology",
            True,
            "require_covisibility_topology must be false",
        ),
    ],
)
def test_contract_rejects_non_reusable_graph_topology(
    tmp_path,
    field,
    replacement_value,
    message,
) -> None:
    contract = _contract(tmp_path)
    graph_config = dict(contract.graph_config)
    graph_config[field] = replacement_value

    with pytest.raises(ValueError, match=message):
        replace(contract, graph_config=graph_config)


def test_changed_data_checkpoint_or_implementation_is_rejected(tmp_path) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    artifact = save_surface_scene_intermediate(value, output)
    bindings = (
        value.contract.source_frames[0].color,
        value.contract.radio_checkpoint,
        value.contract.implementation_sources["support_graph"],
    )
    for binding in bindings:
        with open(binding.path, "rb") as handle:
            original = handle.read()
        with open(binding.path, "ab") as handle:
            handle.write(b"tampered")
        with pytest.raises(ValueError, match="bound source file changed"):
            _trusted_load(output, value, artifact)
        with open(binding.path, "wb") as handle:
            handle.write(original)


def test_source_verification_can_only_be_disabled_with_external_authority(
    tmp_path,
) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    artifact = save_surface_scene_intermediate(value, output)
    color = value.contract.source_frames[0].color
    with open(color.path, "ab") as handle:
        handle.write(b"tampered")

    replay = _trusted_load(
        output,
        value,
        artifact,
        verify_source_files=False,
    )
    assert_exact_surface_scene_replay(value, replay)


def test_wrong_expected_contract_is_rejected(tmp_path) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    artifact = save_surface_scene_intermediate(value, output)
    wrong = replace(value.contract, affinity_dimension=512)

    with pytest.raises(ValueError, match="expected contract"):
        load_surface_scene_intermediate(
            output,
            expected_contract=wrong,
            expected_file_sha256=artifact.file_sha256,
        )


def test_coordinated_metadata_tensor_and_digest_tamper_still_fails_authority(
    tmp_path,
) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    original = save_surface_scene_intermediate(value, output)
    changed_features = value.radio_features.clone()
    changed_features[0, 0] = 0.0
    changed_features[0, 10] = 1.0
    changed_contract = replace(value.contract, radio_resolution=512)
    coordinated = replace(
        value,
        contract=changed_contract,
        radio_features=changed_features,
    )
    changed = save_surface_scene_intermediate(
        coordinated,
        output,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="file SHA-256 differs"):
        load_surface_scene_intermediate(
            output,
            expected_contract=value.contract,
            expected_file_sha256=original.file_sha256,
        )
    with pytest.raises(ValueError, match="expected contract"):
        load_surface_scene_intermediate(
            output,
            expected_contract=value.contract,
            expected_file_sha256=changed.file_sha256,
        )


def test_single_descriptor_rejects_atomic_path_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    replacement = tmp_path / "replacement.pt"
    artifact = save_surface_scene_intermediate(value, output)
    save_surface_scene_intermediate(value, replacement)
    real_loader = scene_module._torch_load_from_handle

    def replace_after_load(handle):
        payload = real_loader(handle)
        os.replace(replacement, output)
        return payload

    monkeypatch.setattr(
        scene_module,
        "_torch_load_from_handle",
        replace_after_load,
    )
    with pytest.raises(ValueError, match="changed during trusted load"):
        _trusted_load(output, value, artifact)


def test_single_descriptor_rejects_in_place_rewrite(
    tmp_path,
    monkeypatch,
) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    artifact = save_surface_scene_intermediate(value, output)
    real_loader = scene_module._torch_load_from_handle

    def rewrite_after_load(handle):
        payload = real_loader(handle)
        with open(output, "r+b") as writer:
            writer.seek(-1, os.SEEK_END)
            original = writer.read(1)
            writer.seek(-1, os.SEEK_END)
            writer.write(bytes([original[0] ^ 0x01]))
            writer.flush()
            os.fsync(writer.fileno())
        return payload

    monkeypatch.setattr(
        scene_module,
        "_torch_load_from_handle",
        rewrite_after_load,
    )
    with pytest.raises(ValueError, match="changed during trusted load"):
        _trusted_load(output, value, artifact)


def test_trusted_load_refuses_final_component_symlink(tmp_path) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    link = tmp_path / "scene-link.pt"
    artifact = save_surface_scene_intermediate(value, output)
    link.symlink_to(output)

    with pytest.raises(ValueError, match="refuse to follow"):
        load_surface_scene_intermediate(
            link,
            expected_contract=value.contract,
            expected_file_sha256=artifact.file_sha256,
        )


def test_no_clobber_preserves_existing_artifact(tmp_path) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    first = save_surface_scene_intermediate(value, output)

    with pytest.raises(FileExistsError, match="already exists"):
        save_surface_scene_intermediate(value, output)
    assert sha256_file(output) == first.file_sha256


def test_no_clobber_is_atomic_against_competing_writer(
    tmp_path,
    monkeypatch,
) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    real_link = scene_module.os.link

    def competing_link(source, destination):
        with open(destination, "wb") as competitor:
            competitor.write(b"competitor-won")
            competitor.flush()
            os.fsync(competitor.fileno())
        return real_link(source, destination)

    monkeypatch.setattr(scene_module.os, "link", competing_link)
    with pytest.raises(FileExistsError, match="already exists"):
        save_surface_scene_intermediate(value, output)

    assert output.read_bytes() == b"competitor-won"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_overwrite_true_replaces_and_remains_trusted(tmp_path) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    output.write_bytes(b"old")

    artifact = save_surface_scene_intermediate(
        value,
        output,
        overwrite=True,
    )
    replay = _trusted_load(output, value, artifact)
    assert_exact_surface_scene_replay(value, replay)


def test_successful_publish_fsyncs_parent_for_both_modes(
    tmp_path,
    monkeypatch,
) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "cache" / "scene.pt"
    calls = []
    monkeypatch.setattr(
        scene_module,
        "_fsync_directory",
        lambda path: calls.append(path),
    )

    save_surface_scene_intermediate(value, output)
    save_surface_scene_intermediate(value, output, overwrite=True)

    assert calls == [output.parent.absolute(), output.parent.absolute()]


def test_tensor_digest_binds_dtype_shape_and_bytes() -> None:
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.double())
    assert tensor_sha256(value) != tensor_sha256(value.reshape(2, 1))
    changed = value.clone()
    changed[0, 0] += 1
    assert tensor_sha256(value) != tensor_sha256(changed)


def test_internal_tensor_tamper_is_rejected_after_external_rebinding(
    tmp_path,
) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    save_surface_scene_intermediate(value, output)
    payload = _torch_load(output)
    payload["tensors"]["xyz"][0, 0] = 9.0
    torch.save(payload, output)

    with pytest.raises(ValueError, match="tensor digest differs"):
        load_surface_scene_intermediate(
            output,
            expected_contract=value.contract,
            expected_file_sha256=sha256_file(output),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "radio_features",
            torch.zeros(3, 1280, dtype=torch.float64),
            "dtype",
        ),
        (
            "radio_features",
            torch.zeros(3, 1280, dtype=torch.float32),
            "L2 normalized",
        ),
        (
            "xyz",
            torch.tensor(
                [[float("nan"), 0, 0], [0.1, 0, 0], [0.2, 0, 0]],
                dtype=torch.float32,
            ),
            "finite",
        ),
        (
            "geometric_reliability",
            torch.tensor([0.5, 1.01, 0.9], dtype=torch.float32),
            r"\[0,1\]",
        ),
    ],
)
def test_scientific_tensor_validation_rejects_invalid_values(
    tmp_path,
    field,
    replacement,
    message,
) -> None:
    value = _intermediate(tmp_path)
    with pytest.raises((TypeError, ValueError), match=message):
        replace(value, **{field: replacement})


def _channels(edge_count, **updates):
    values = {
        "geometry": torch.full((edge_count,), 0.9, dtype=torch.float32),
        "appearance": torch.full((edge_count,), 0.8, dtype=torch.float32),
        "boundary": torch.ones(edge_count, dtype=torch.float32),
    }
    values.update(updates)
    return values


def test_graph_rejects_wrong_channel_authority_and_ranges(tmp_path) -> None:
    value = _intermediate(tmp_path)
    missing = dict(value.graph.edge_channels)
    missing.pop("geometry")
    with pytest.raises(ValueError, match="graph edge channels keys differ"):
        replace(value, graph=_graph(channels=missing))

    extra = dict(value.graph.edge_channels)
    extra["normal"] = torch.ones(4)
    with pytest.raises(ValueError, match="graph edge channels keys differ"):
        replace(value, graph=_graph(channels=extra))

    raw = value.graph.raw_affinity.clone()
    raw[0] = raw[1] = 1.01
    with pytest.raises(ValueError, match=r"raw_affinity must lie in \[0,1\]"):
        replace(value, graph=_graph(raw_affinity=raw))

    channels = dict(value.graph.edge_channels)
    channels["appearance"] = channels["appearance"].clone()
    channels["appearance"][0] = channels["appearance"][1] = 1.01
    with pytest.raises(ValueError, match=r"appearance must lie in \[0,1\]"):
        replace(value, graph=_graph(channels=channels))


def test_active_support_graph_builder_output_is_accepted(tmp_path) -> None:
    value = _intermediate(tmp_path)
    appearance = torch.tensor(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
        dtype=torch.float32,
    )
    boundary = torch.tensor(
        [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
        dtype=torch.float32,
    )
    graph = build_primitive_support_graph(
        value.xyz,
        appearance_features=appearance,
        boundary_features=boundary,
        config=SupportGraphConfig(**dict(value.contract.graph_config)),
    )

    accepted = replace(value, graph=graph)

    assert set(accepted.graph.edge_channels) == EXPECTED_EDGE_CHANNELS


def test_mutual_one_neighbor_builder_is_rejected_at_contract_stage(
    tmp_path,
) -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    features = torch.eye(3, dtype=torch.float32)
    graph_config = SupportGraphConfig(
        neighbors=1,
        topology_mode="mutual_knn",
    )
    graph = build_primitive_support_graph(
        xyz,
        appearance_features=features,
        boundary_features=features,
        config=graph_config,
    )
    out_degree = torch.bincount(graph.edge_index[0], minlength=3)
    assert out_degree.tolist() == [1, 1, 0]

    with pytest.raises(
        ValueError,
        match="topology_mode must be symmetric_union",
    ):
        replace(
            _contract(tmp_path),
            graph_config=default_graph_config_dict(graph_config),
        )


@pytest.mark.parametrize("num_nodes", [1, 2, 17])
def test_active_surface_contract_real_builder_graph_is_accepted(
    tmp_path,
    num_nodes,
) -> None:
    active_graph_config = SurfaceRegionContractV2().graph_config()
    contract = replace(
        _contract(tmp_path),
        graph_config=default_graph_config_dict(active_graph_config),
    )
    xyz = torch.zeros(num_nodes, 3, dtype=torch.float32)
    xyz[:, 0] = torch.arange(num_nodes, dtype=torch.float32) * 0.05
    affinity_features = torch.zeros(num_nodes, 256, dtype=torch.float32)
    affinity_features[
        torch.arange(num_nodes),
        torch.arange(num_nodes),
    ] = 1.0
    graph = build_primitive_support_graph(
        xyz,
        appearance_features=affinity_features,
        boundary_features=torch.roll(affinity_features, 1, dims=1),
        config=active_graph_config,
    )
    radio_features = torch.zeros(num_nodes, 1280, dtype=torch.float32)
    radio_features[
        torch.arange(num_nodes),
        torch.arange(num_nodes),
    ] = 1.0

    accepted = SurfaceSceneIntermediate(
        contract=contract,
        xyz=xyz,
        radio_features=radio_features,
        geometric_reliability=torch.ones(num_nodes, dtype=torch.float32),
        graph=graph,
    )

    assert accepted.graph.num_nodes == num_nodes
    assert set(accepted.graph.edge_channels) == EXPECTED_EDGE_CHANNELS


def test_graph_rejects_self_duplicate_and_missing_reverse_edges(tmp_path) -> None:
    value = _intermediate(tmp_path)
    self_edges = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 1, 2]],
        dtype=torch.int64,
    )
    raw = torch.full((6,), 0.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="self edges"):
        replace(
            value,
            graph=_graph(
                edge_index=self_edges,
                raw_affinity=raw,
                channels=_channels(6),
            ),
        )

    duplicate_edges = torch.tensor(
        [[0, 0, 1, 1, 1, 2], [1, 1, 0, 0, 2, 1]],
        dtype=torch.int64,
    )
    with pytest.raises(ValueError, match="duplicate directed"):
        replace(
            value,
            graph=_graph(
                edge_index=duplicate_edges,
                raw_affinity=raw,
                channels=_channels(6),
            ),
        )

    missing_reverse = torch.tensor(
        [[0, 1, 2], [1, 2, 1]],
        dtype=torch.int64,
    )
    raw3 = torch.full((3,), 0.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="lacks a unique reverse"):
        replace(
            value,
            graph=_graph(
                edge_index=missing_reverse,
                raw_affinity=raw3,
                channels=_channels(3),
            ),
        )


def test_graph_rejects_asymmetric_values_wrong_weights_and_isolation(
    tmp_path,
) -> None:
    value = _intermediate(tmp_path)
    raw = value.graph.raw_affinity.clone()
    raw[1] -= 0.01
    with pytest.raises(ValueError, match="reverse raw affinities"):
        replace(value, graph=_graph(raw_affinity=raw))

    channels = dict(value.graph.edge_channels)
    channels["boundary"] = channels["boundary"].clone()
    channels["boundary"][1] = 0.9
    with pytest.raises(ValueError, match="reverse boundary affinities"):
        replace(value, graph=_graph(channels=channels))

    with pytest.raises(ValueError, match="raw affinity / row sum"):
        replace(
            value,
            graph=_graph(edge_weight=torch.full((4,), 0.5)),
        )

    isolated_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    raw2 = torch.full((2,), 0.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="every graph node"):
        replace(
            value,
            graph=_graph(
                edge_index=isolated_edges,
                raw_affinity=raw2,
                channels=_channels(2),
            ),
        )


def test_loader_rejects_boolean_num_nodes_and_wrong_raw_dtype(tmp_path) -> None:
    value = _intermediate(tmp_path)
    output = tmp_path / "scene.pt"
    save_surface_scene_intermediate(value, output)
    payload = _torch_load(output)
    payload["tensors"]["graph"]["num_nodes"] = True
    torch.save(payload, output)
    with pytest.raises(ValueError, match="num_nodes is invalid"):
        load_surface_scene_intermediate(
            output,
            expected_contract=value.contract,
            expected_file_sha256=sha256_file(output),
        )

    save_surface_scene_intermediate(value, output, overwrite=True)
    payload = _torch_load(output)
    payload["tensors"]["graph"]["edge_index"] = payload["tensors"][
        "graph"
    ]["edge_index"].to(torch.int32)
    torch.save(payload, output)
    with pytest.raises(TypeError, match="torch.int64"):
        load_surface_scene_intermediate(
            output,
            expected_contract=value.contract,
            expected_file_sha256=sha256_file(output),
        )


def test_exact_replay_helper_detects_scientific_or_contract_difference(
    tmp_path,
) -> None:
    fresh = _intermediate(tmp_path)
    replay_features = fresh.radio_features.clone()
    replay_features[0, 0] = 0.0
    replay_features[0, 10] = 1.0
    changed = replace(fresh, radio_features=replay_features)
    with pytest.raises(AssertionError, match="radio_features"):
        assert_exact_surface_scene_replay(fresh, changed)

    changed_contract = replace(fresh.contract, adaptor_batch_size=128)
    with pytest.raises(AssertionError, match="contracts differ"):
        assert_exact_surface_scene_replay(
            fresh,
            replace(fresh, contract=changed_contract),
        )
