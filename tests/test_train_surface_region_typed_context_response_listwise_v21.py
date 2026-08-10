from __future__ import annotations

import torch
import torch.nn.functional as F

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.losses.source_global_response_listwise_loss_v21 import (
    CANONICAL_NEGATIVE_MODEL,
    FrozenCanonicalNegativeBank,
    FrozenCompositionalGenericBank,
)
from radio_gs.losses.source_typed_text_relation_loss_v21 import (
    FrozenTypedTextRelationAuthority,
)
from radio_gs.scripts import (
    train_surface_region_typed_context_response_listwise_v21 as trainer,
)
from tests.test_train_surface_region_typed_context_response_listwise_v2 import (
    _fixture,
)


def test_complete_scene_v21_adapter_is_finite_and_parameter_free() -> None:
    normalization, scene, text, authority, model = _fixture()
    negatives = torch.zeros(4, 1536, dtype=torch.float32)
    negatives[:, -1] = 1.0
    bank = FrozenCanonicalNegativeBank(
        embeddings=negatives,
        file_sha256="7" * 64,
        embedding_tensor_sha256=tensor_sha256(negatives),
        model_id=CANONICAL_NEGATIVE_MODEL,
    )
    total, metrics = trainer.complete_scene_objective_v21(
        model,
        scene,
        normalization,
        text,
        authority,
        bank,
        torch.device("cpu"),
    )
    assert total.ndim == 0 and bool(torch.isfinite(total))
    assert metrics["complete_canonical_rows"] == 4
    assert metrics["fallback_bitwise_accepted_v2_e0"] is True
    assert metrics["response_generic_bank_components"] == 1
    total.backward()
    assert model.residual_projection.weight.grad is not None
    assert float(model.residual_projection.weight.grad.abs().sum()) > 0


def test_adapter_accepts_audited_external_routing_and_filters_training_pairs() -> None:
    normalization, scene, text, authority, model = _fixture()
    negatives = torch.zeros(4, 1536, dtype=torch.float32)
    negatives[:, -1] = 1.0
    bank = FrozenCanonicalNegativeBank(
        embeddings=negatives,
        file_sha256="7" * 64,
        embedding_tensor_sha256=tensor_sha256(negatives),
        model_id=CANONICAL_NEGATIVE_MODEL,
    )
    declared = torch.tensor([True, True, False, True])
    effective_ood = torch.tensor([False, True, True, False])
    active = torch.tensor([True, False, False, True])
    _, metrics = trainer.complete_scene_objective_v21(
        model,
        scene,
        normalization,
        text,
        authority,
        bank,
        torch.device("cpu"),
        exclude_both_immutable_pairs=True,
        routing_masks=(declared, effective_ood, active),
    )
    assert metrics["active_rows"] == 2
    assert metrics["response_authority_hard_negative_pairs"] == 4
    assert metrics["response_objective_hard_negative_pairs"] == 3
    assert metrics["response_both_immutable_pairs_excluded"] == 1
    contract = trainer.integration_contract()
    assert contract["new_learnable_parameters"] is False
    assert contract["optimizer_constructed_by_adapter"] is False
    assert contract["benchmark_execution_supported"] is False
    assert contract["typed_relation_authority_supported"] is True
    assert contract["typed_relation_runtime_query_strings_consumed"] is False


def test_adapter_has_no_benchmark_cli_or_runtime_entrypoint() -> None:
    assert not hasattr(trainer, "build_parser")
    assert not hasattr(trainer, "train")
    assert trainer.source_access()["benchmark_text_queries_opened"] is False


def test_complete_scene_adapter_consumes_sha_bound_typed_relations() -> None:
    normalization, scene, text, response_authority, model = _fixture()
    negatives = torch.zeros(4, 1536, dtype=torch.float32)
    negatives[:, -1] = 1.0
    canonical = FrozenCanonicalNegativeBank(
        embeddings=negatives,
        file_sha256="7" * 64,
        embedding_tensor_sha256=tensor_sha256(negatives),
        model_id=CANONICAL_NEGATIVE_MODEL,
    )
    generator = torch.Generator().manual_seed(91)
    specifications = (
        ("synonym_relation", 0.20, "1" * 64),
        ("lexical_sibling_relation", 0.20, "2" * 64),
        ("counterfactual_attributes", 0.30, "3" * 64),
        ("high_precision_part_of", 0.05, "4" * 64),
    )
    components = []
    component_records = {
        "primary": {
            "path": "/primary.pt",
            "sha256": response_authority.fit_text_bank_file_sha256,
            "embedding_tensor_sha256": (
                response_authority.fit_text_embedding_tensor_sha256
            ),
            "query_rows": int(text.shape[0]),
        }
    }
    for component_id, loss_weight, file_sha in specifications:
        embeddings = F.normalize(torch.randn(2, 1536, generator=generator), dim=-1)
        embedding_sha = tensor_sha256(embeddings)
        components.append(
            FrozenCompositionalGenericBank(
                component_id=component_id,
                embeddings=embeddings,
                file_sha256=file_sha,
                embedding_tensor_sha256=embedding_sha,
                model_id=CANONICAL_NEGATIVE_MODEL,
                query_rows=2,
                loss_weight=loss_weight,
            )
        )
        component_records[component_id] = {
            "path": f"/{component_id}.pt",
            "sha256": file_sha,
            "embedding_tensor_sha256": embedding_sha,
            "query_rows": 2,
        }
    relation_authority = FrozenTypedTextRelationAuthority(
        file_sha256="5" * 64,
        content_authority_sha256="6" * 64,
        source_sha256="8" * 64,
        components=component_records,
        synonym_left_primary_indices=torch.tensor([0, 1]),
        synonym_right_component_indices=torch.tensor([0, 1]),
        sibling_left_primary_indices=torch.tensor([0]),
        sibling_right_primary_indices=torch.tensor([1]),
    )
    total, metrics = trainer.complete_scene_objective_v21(
        model,
        scene,
        normalization,
        text,
        response_authority,
        canonical,
        torch.device("cpu"),
        compositional_banks=components,
        relation_authority=relation_authority,
    )
    assert total.ndim == 0 and bool(torch.isfinite(total))
    assert metrics["response_typed_relation_active"] == 1
    assert metrics["response_synonym_pair_region_units"] == 8
    assert (
        metrics["response_sibling_left_dominant_units"]
        + metrics["response_sibling_right_dominant_units"]
        == 4
    )
    total.backward()
    assert model.residual_projection.weight.grad is not None
    assert float(model.residual_projection.weight.grad.abs().sum()) > 0
