import hashlib
import json

import pytest
import torch

from radio_gs.field.factorized_radio_contract import (
    CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
)
from radio_gs.querying.support_solver import (
    SupportGraphConfig,
    build_primitive_support_graph,
    graph_for_query_intent,
)
from radio_gs.querying.query_spec import QueryIntent
from radio_gs.scripts.build_canonical_support_graph import (
    build,
    capability_affinity_features,
    deterministic_feature_hash,
    estimate_unoriented_local_surface_normals,
    load_covisibility_observations,
    visibility_from_registration_responsibility,
)


def test_support_graph_builder_no_clobber_precedes_loading(tmp_path):
    from argparse import Namespace

    output = tmp_path / "graph.pt"
    output.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="refuses to clobber"):
        build(Namespace(output=str(output)))


def test_deterministic_feature_hash_is_repeatable_and_normalized():
    features = torch.arange(60, dtype=torch.float32).reshape(6, 10) + 1.0
    first = deterministic_feature_hash(features, 7, batch_size=2)
    second = deterministic_feature_hash(features, 7, batch_size=4)

    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(dim=-1), torch.ones(6), atol=1e-6)


def test_exact_capability_affinity_keeps_the_official_rows_without_hashing():
    """The high-fidelity graph variant must not silently reduce DINO/SAM.

    This is a graph-construction choice only: the selected rows are exactly
    the canonical official capability tensors, so it cannot introduce query
    or evaluator state.
    """

    appearance = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    boundary = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100.0
    selected = torch.tensor([3, 1])
    result_appearance, result_boundary, audit = capability_affinity_features(
        {
            "appearance_dino_v3": appearance,
            "boundary_sam3": boundary,
        },
        selected,
        mode="exact_official_capability",
        affinity_dim=7,
        hash_batch_size=2,
    )

    torch.testing.assert_close(result_appearance, appearance[selected])
    torch.testing.assert_close(result_boundary, boundary[selected])
    assert audit == {
        "mode": "exact_official_capability",
        "appearance_dim": 6,
        "boundary_dim": 3,
        "query_independent": True,
    }

    # The graph builder accepts these native-dimensional rows directly. The
    # CPU backend is the deterministic test configuration; production may
    # choose CUDA only for the same chunked cosine computation.
    graph = build_primitive_support_graph(
        torch.tensor([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        appearance_features=result_appearance,
        boundary_features=result_boundary,
        config=SupportGraphConfig(neighbors=1),
        feature_affinity_device="cpu",
    )
    assert set(graph.edge_channels) == {"geometry", "appearance", "boundary"}
    assert torch.isfinite(graph.edge_channels["appearance"]).all()
    assert torch.isfinite(graph.edge_channels["boundary"]).all()


def test_exact_capability_graph_composes_with_manifold_and_covisibility_relations():
    """The high-fidelity graph may combine only field-side relations.

    This is the compositional promotion candidate used after the isolated
    graph rows: native official DINO/SAM affinities, local tangent continuity,
    and MPR view co-visibility all share the same frozen primitive domain.
    The test deliberately supplies no query/object/label state.
    """

    xyz = torch.tensor([[float(x), float(y), 0.0] for y in range(3) for x in range(3)])
    appearance = torch.arange(72, dtype=torch.float32).reshape(9, 8) + 1.0
    boundary = torch.arange(45, dtype=torch.float32).reshape(9, 5) + 3.0
    exact_appearance, exact_boundary, audit = capability_affinity_features(
        {
            "appearance_dino_v3": appearance,
            "boundary_sam3": boundary,
        },
        torch.arange(9),
        mode="exact_official_capability",
        affinity_dim=256,
        hash_batch_size=4,
    )
    normals, normal_reliability = estimate_unoriented_local_surface_normals(
        xyz, neighbors=8, batch_size=3
    )
    view_observations = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, False, False],
            [False, True, True],
            [False, True, True],
            [False, False, True],
            [True, False, True],
            [True, False, True],
            [False, False, True],
        ]
    )
    graph = build_primitive_support_graph(
        xyz,
        appearance_features=exact_appearance,
        boundary_features=exact_boundary,
        normals=normals,
        normal_reliability=normal_reliability,
        view_observations=view_observations,
        config=SupportGraphConfig(
            neighbors=4,
            surface_tangent_relation=True,
            covisibility_weight=0.25,
        ),
        feature_affinity_device="cpu",
    )

    assert audit["mode"] == "exact_official_capability"
    assert set(graph.edge_channels) == {
        "geometry",
        "appearance",
        "boundary",
        "normal",
        "surface_tangent",
        "covisibility",
    }
    mixed = graph_for_query_intent(graph, QueryIntent.INSTANCE, policy="typed")
    assert torch.isfinite(mixed.edge_weight).all()
    assert torch.isfinite(mixed.raw_affinity).all()


def test_local_surface_normal_estimate_is_sign_agnostic_and_planarity_weighted():
    grid = torch.tensor([[float(x), float(y), 0.0] for y in range(3) for x in range(3)])
    normals, reliability = estimate_unoriented_local_surface_normals(
        grid, neighbors=8, batch_size=3
    )

    assert normals.shape == grid.shape
    assert reliability.shape == (len(grid),)
    assert torch.all(normals[:, 2].abs() > 0.99)
    # Boundary neighborhoods are spatially anisotropic and therefore receive
    # lower (but nonzero) confidence; the fully surrounded centre is planar.
    assert torch.all(reliability > 0.20)
    assert reliability[4] > 0.95


def test_registration_responsibility_visibility_keeps_only_capability_valid_rows():
    payload = {
        "schema_version": 1,
        "assignments": [
            {"gaussian_ids": torch.tensor([0, 1, 1, 3])},
            {"gaussian_ids": torch.tensor([1, 2])},
            {"gaussian_ids": torch.tensor([0, 2, 3])},
        ],
    }
    visible, audit = visibility_from_registration_responsibility(
        payload,
        num_global_rows=4,
        global_rows=torch.tensor([0, 2, 3]),
    )
    assert visible.dtype == torch.bool
    assert visible.tolist() == [
        [True, False, True],
        [False, True, True],
        [True, False, True],
    ]
    assert audit == {
        "num_views": 3,
        "valid_primitives_with_any_view": 3,
        "registered_global_rows_before_capability_filter": 8,
    }


def test_covisibility_sidecar_is_digest_bound_to_the_capability_mpr(tmp_path):
    responsibility = tmp_path / "responsibility.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": {
                "xyz_sha256": "geometry-digest",
                "selected_dataset_indices": [3, 7],
            },
            "assignments": [
                {"gaussian_ids": torch.tensor([0, 1])},
                {"gaussian_ids": torch.tensor([1, 2])},
            ],
        },
        responsibility,
    )
    digest = hashlib.sha256(responsibility.read_bytes()).hexdigest()
    mpr = tmp_path / "raw_radio_mpr.pt"
    mpr.touch()
    (tmp_path / "raw_radio_mpr.pt.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "registration_responsibility_cache_sha256": digest,
                    "xyz_sha256": "geometry-digest",
                }
            }
        ),
        encoding="utf-8",
    )
    visible, audit = load_covisibility_observations(
        {"mpr_cache": str(mpr)},
        responsibility_cache=responsibility,
        num_global_rows=3,
        global_rows=torch.tensor([0, 2]),
    )
    assert visible.tolist() == [[True, False], [False, True]]
    assert audit["responsibility_cache_sha256"] == digest
    assert audit["raw_mpr_xyz_sha256"] == "geometry-digest"

    (tmp_path / "raw_radio_mpr.pt.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "registration_responsibility_cache_sha256": "wrong",
                    "xyz_sha256": "geometry-digest",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest"):
        load_covisibility_observations(
            {"mpr_cache": str(mpr)},
            responsibility_cache=responsibility,
            num_global_rows=3,
            global_rows=torch.tensor([0, 2]),
        )


def test_factorized_covisibility_reuses_validated_top_level_geometry(tmp_path):
    geometry_sha = "a" * 64
    responsibility = tmp_path / "factorized-responsibility.pt"
    torch.save(
        {
            "schema_version": 1,
            "metadata": {
                "xyz_sha256": geometry_sha,
                "selected_dataset_indices": [0, 1],
            },
            "assignments": [
                {"gaussian_ids": torch.tensor([0, 1])},
                {"gaussian_ids": torch.tensor([1, 2])},
            ],
        },
        responsibility,
    )
    digest = hashlib.sha256(responsibility.read_bytes()).hexdigest()
    mpr = tmp_path / "factorized-radio.pt"
    mpr.touch()
    (tmp_path / "factorized-radio.pt.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "registration_responsibility_cache_sha256": digest,
                }
            }
        ),
        encoding="utf-8",
    )
    capability_metadata = {
        "mpr_cache": str(mpr),
        "field_checkpoint_schema_version": 2,
        "field_checkpoint_contract": CANONICAL_FACTORIZED_RADIO_CHECKPOINT_CONTRACT,
        "registration_responsibility_cache_sha256": digest,
        "mpr_geometry_fingerprint": {
            "num_gaussians": 3,
            "xyz_sha256": geometry_sha,
        },
    }
    visible, audit = load_covisibility_observations(
        capability_metadata,
        responsibility_cache=responsibility,
        num_global_rows=3,
        global_rows=torch.tensor([0, 2]),
    )
    assert visible.tolist() == [[True, False], [False, True]]
    assert audit["raw_mpr_xyz_sha256"] == geometry_sha

    (tmp_path / "factorized-radio.pt.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "registration_responsibility_cache_sha256": "0" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar responsibility digest"):
        load_covisibility_observations(
            capability_metadata,
            responsibility_cache=responsibility,
            num_global_rows=3,
            global_rows=torch.tensor([0, 2]),
        )

    (tmp_path / "factorized-radio.pt.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "registration_responsibility_cache_sha256": digest,
                }
            }
        ),
        encoding="utf-8",
    )

    corrupted = dict(capability_metadata)
    corrupted["field_checkpoint_contract"] = "wrong"
    with pytest.raises(ValueError, match="exact checkpoint lineage"):
        load_covisibility_observations(
            corrupted,
            responsibility_cache=responsibility,
            num_global_rows=3,
            global_rows=torch.tensor([0, 2]),
        )
