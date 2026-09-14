"""Build the query-free second-level LERF object-hypothesis memory."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.scripts.extract_official_crop_summary_teacher import _atomic_torch_save
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.contracts.lerf_fragment_surface_memory import validate_memory
from radio_gs.v4.evaluation.lerf_fragment_ceiling import _load_fragment_memory
from radio_gs.v4.evaluation.lerf_object_ceiling import _load_state
from radio_gs.v4.evaluation.lerf_text_cold_evaluator import _load_fragment_prototypes
from radio_gs.v4.object_memory.fragment_to_object import (
    FragmentAssociationConfiguration,
    build_fragment_object_hypotheses,
)
from radio_gs.v4.object_memory.learned_fragment_affinity import checkpoint_model


SCHEMA = "radio_gs.surface_object_memory_v4.lerf_object_hypotheses.v1"


def _select_hypothesis_prototypes(
    appearance: torch.Tensor,
    fragment_assignment: torch.Tensor,
    quality: torch.Tensor,
    *,
    maximum_prototypes: int,
    fragment_view_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    descriptors = torch.as_tensor(appearance, dtype=torch.float32).cpu()
    assignment = torch.as_tensor(fragment_assignment, dtype=torch.float32).cpu()
    quality = torch.as_tensor(quality, dtype=torch.float32).cpu()
    if descriptors.ndim != 3 or assignment.shape[0] != descriptors.shape[0]:
        raise ValueError("fragment appearance and object assignments do not align")
    if quality.shape != (descriptors.shape[0],) or maximum_prototypes <= 0:
        raise ValueError("prototype quality or capacity is invalid")
    views = None if fragment_view_index is None else torch.as_tensor(fragment_view_index, dtype=torch.long).cpu()
    if views is not None and (views.shape != quality.shape or bool((views < 0).any())):
        raise ValueError("prototype source views do not align")
    if not bool(torch.isfinite(descriptors).all()) or bool((descriptors.norm(dim=-1) <= 0).any()):
        raise ValueError("source prototypes must be finite and nonzero")
    hypothesis_count = assignment.shape[1]
    output = torch.zeros(
        hypothesis_count, maximum_prototypes, descriptors.shape[2], dtype=torch.float32
    )
    fragment_ids = torch.full((hypothesis_count, maximum_prototypes), -1, dtype=torch.long)
    prototype_kind = torch.full_like(fragment_ids, -1)
    valid = torch.zeros_like(fragment_ids, dtype=torch.bool)
    for hypothesis_id in range(hypothesis_count):
        score = assignment[:, hypothesis_id] * quality
        ranked = torch.argsort(score, descending=True, stable=True)
        if views is not None:
            # Spend the limited slots on distinct observations first. Nested
            # masks from one photograph are not independent visual evidence.
            first, rest, seen = [], [], set()
            for fragment_id in ranked.tolist():
                if float(score[fragment_id]) <= 0:
                    break
                view = int(views[fragment_id])
                (rest if view in seen else first).append(fragment_id)
                seen.add(view)
            ranked = torch.tensor(first + rest, dtype=torch.long)
        slot = 0
        for fragment_id in ranked.tolist():
            if float(score[fragment_id]) <= 0:
                break
            for kind in range(descriptors.shape[1]):
                output[hypothesis_id, slot] = descriptors[fragment_id, kind]
                fragment_ids[hypothesis_id, slot] = fragment_id
                prototype_kind[hypothesis_id, slot] = kind
                valid[hypothesis_id, slot] = True
                slot += 1
                if slot == maximum_prototypes:
                    break
            if slot == maximum_prototypes:
                break
    if not bool(valid.any(-1).all()):
        raise ValueError("every object hypothesis needs at least one source prototype")
    result = {
        "object_prototype_descriptors": output,
        "object_prototype_fragment_ids": fragment_ids,
        "object_prototype_kind": prototype_kind,
        "object_prototype_valid": valid,
    }
    if views is not None:
        result["object_prototype_view_ids"] = torch.where(valid, views[fragment_ids.clamp_min(0)], -1)
    return result


def validate_hypothesis_memory(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("LERF object-hypothesis memory schema differs")
    assignment = torch.as_tensor(payload.get("fragment_assignment"), dtype=torch.float32).cpu()
    null = torch.as_tensor(payload.get("fragment_null_probability"), dtype=torch.float32).cpu()
    membership = torch.as_tensor(payload.get("observed_membership"), dtype=torch.float32).cpu()
    if assignment.ndim != 2 or min(assignment.shape) <= 0:
        raise ValueError("fragment assignments must have shape [F,K]")
    if null.shape != (assignment.shape[0],) or membership.shape != (
        int(payload.get("surface_element_count", -1)), assignment.shape[1]
    ):
        raise ValueError("hypothesis assignment and surface axes do not align")
    if not all(bool(torch.isfinite(value).all()) for value in (assignment, null, membership)):
        raise ValueError("hypothesis memory contains non-finite probabilities")
    if any(bool(((value < 0) | (value > 1)).any()) for value in (assignment, null, membership)):
        raise ValueError("hypothesis probabilities must lie in [0,1]")
    completed = payload.get("completed_membership")
    if completed is not None:
        completed = torch.as_tensor(completed, dtype=torch.float32).cpu()
        if completed.shape != membership.shape or not bool(torch.isfinite(completed).all()):
            raise ValueError("completed hypothesis membership must align and be finite")
        if bool(((completed < 0) | (completed > 1)).any()):
            raise ValueError("completed hypothesis membership must lie in [0,1]")
        if bool((completed + 1e-4 < membership).any()):
            raise ValueError("completion must not erase observed hypothesis evidence")
    if not torch.allclose(assignment.sum(-1) + null, torch.ones_like(null), atol=2e-3, rtol=0):
        raise ValueError("fragment assignment and null probability must form a simplex")
    if int(payload.get("fragment_count", -1)) != assignment.shape[0]:
        raise ValueError("fragment count differs from the assignment axis")
    groups = payload.get(
        "hypothesis_seed_fragments", payload.get("hypothesis_root_fragments")
    )
    if not isinstance(groups, list) or len(groups) != assignment.shape[1]:
        raise ValueError("root-fragment groups differ from the hypothesis axis")
    representatives = payload.get("hypothesis_representative_fragments")
    if representatives is not None:
        representatives = torch.as_tensor(representatives, dtype=torch.long).cpu()
        if representatives.shape != (assignment.shape[1],) or bool(
            ((representatives < 0) | (representatives >= assignment.shape[0])).any()
        ):
            raise ValueError("hypothesis representative fragments are invalid")
    if "object_prototype_descriptors" in payload:
        descriptors = torch.as_tensor(
            payload["object_prototype_descriptors"], dtype=torch.float32
        ).cpu()
        valid = torch.as_tensor(payload.get("object_prototype_valid"), dtype=torch.bool).cpu()
        fragment_ids = torch.as_tensor(
            payload.get("object_prototype_fragment_ids"), dtype=torch.long
        ).cpu()
        kind = torch.as_tensor(payload.get("object_prototype_kind"), dtype=torch.long).cpu()
        if descriptors.ndim != 3 or descriptors.shape[0] != assignment.shape[1]:
            raise ValueError("object prototypes do not align with hypotheses")
        if any(value.shape != descriptors.shape[:2] for value in (valid, fragment_ids, kind)):
            raise ValueError("object prototype metadata does not align")
        if not bool(valid.any(-1).all()) or bool(((fragment_ids[valid] < 0) | (fragment_ids[valid] >= assignment.shape[0])).any()):
            raise ValueError("every hypothesis needs valid source prototype receipts")
        if not bool(torch.isfinite(descriptors[valid]).all()) or bool((descriptors[valid].norm(dim=-1) <= 0).any()):
            raise ValueError("valid object prototypes must be finite and nonzero")
        if bool((kind[valid] < 0).any()):
            raise ValueError("valid object prototypes need crop-kind receipts")
        if "object_prototype_view_ids" in payload:
            views = torch.as_tensor(payload["object_prototype_view_ids"], dtype=torch.long)
            if views.shape != valid.shape or bool((views[valid] < 0).any()):
                raise ValueError("object prototype source-view receipts are invalid")
            for fragment_id in torch.unique(fragment_ids[valid]):
                if torch.unique(views[valid & (fragment_ids == fragment_id)]).numel() != 1:
                    raise ValueError("one source fragment cannot belong to multiple views")
    policy = payload.get("information_policy", {})
    forbidden = (
        "benchmark_labels_opened",
        "benchmark_masks_opened",
        "text_queries_opened",
        "target_rgb_opened",
    )
    if any(policy.get(key) is not False for key in forbidden):
        raise ValueError("hypothesis construction opened a forbidden information channel")
    result = dict(payload)
    result.update({
        "fragment_assignment": assignment,
        "fragment_null_probability": null,
        "observed_membership": membership,
        "hypothesis_seed_fragments": groups,
    })
    if completed is not None:
        result["completed_membership"] = completed
    if representatives is not None:
        result["hypothesis_representative_fragments"] = representatives
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu_threads <= 0:
        raise ValueError("cpu thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    state_path = Path(args.scene_state)
    state = _load_state(state_path, expected_sha256=args.expected_scene_state_sha256)
    fragment_path = Path(args.fragment_memory)
    fragment_memory = _load_fragment_memory(
        fragment_path,
        expected_sha256=args.expected_fragment_memory_sha256,
        scene_state_sha256=state["scene_state_sha256"],
    )
    fragment_memory = validate_memory(fragment_memory)
    if state["scene_label"] != args.scene_label or fragment_memory["scene_label"] != args.scene_label:
        raise ValueError("scene labels differ across object-hypothesis inputs")

    if args.appearance_source == "none":
        appearance = None
        appearance_audit = {
            "source": "none",
            "prototype_count_per_fragment": 0,
            "valid_for_text_query": False,
        }
    elif args.appearance_source == "legacy_state_prototypes":
        appearance = torch.as_tensor(state["prototype_descriptors"], dtype=torch.float32)
        if appearance.shape[0] != fragment_memory["fragment_positive"].shape[0]:
            raise ValueError("legacy appearance prototypes do not align with fragments")
        appearance = appearance[:, None]
        appearance_audit = {
            "source": "legacy_state_prototypes_appearance_only",
            "prototype_count_per_fragment": 1,
            "valid_for_text_query": False,
            "warning": "per-spatial SummaryHead semantics are forbidden for text; reused only as an appearance affinity ablation",
        }
    elif args.appearance_source == "fragment_language_manifest":
        if not args.fragment_language_manifest:
            raise ValueError("fragment-language appearance requires its manifest")
        flat, _legacy_token_ids, semantic_audit = _load_fragment_prototypes(
            Path(args.fragment_language_manifest), state=state
        )
        fragment_count = fragment_memory["fragment_positive"].shape[0]
        if flat.shape[0] != 2 * fragment_count:
            raise ValueError("fragment language prototypes do not align with fragment memory")
        appearance = torch.stack((flat[:fragment_count], flat[fragment_count:]), dim=1)
        appearance_audit = {
            **semantic_audit,
            "source": "official_masked_and_context_crop_summary_prototypes",
            "prototype_count_per_fragment": 2,
            "valid_for_text_query": True,
        }
    else:
        raise ValueError("unsupported fragment appearance source")

    learned_affinity_model = None
    learned_affinity_audit = None
    merge_affinity = float(args.merge_affinity)
    if args.learned_affinity_checkpoint:
        if not args.expected_learned_affinity_checkpoint_sha256:
            raise ValueError("learned affinity checkpoint requires an expected SHA256")
        affinity_path = Path(args.learned_affinity_checkpoint).resolve(strict=True)
        affinity_sha = sha256_file(affinity_path)
        if affinity_sha != args.expected_learned_affinity_checkpoint_sha256:
            raise ValueError("learned affinity checkpoint SHA256 differs")
        affinity_checkpoint = torch.load(
            affinity_path, map_location="cpu", weights_only=False
        )
        learned_affinity_model = checkpoint_model(affinity_checkpoint)
        merge_affinity = float(affinity_checkpoint["decision_threshold"])
        learned_affinity_audit = {
            "checkpoint": str(affinity_path),
            "checkpoint_sha256": affinity_sha,
            "training_scene_ids": list(affinity_checkpoint["training_scene_ids"]),
            "validation_scene_ids": list(affinity_checkpoint["validation_scene_ids"]),
            "decision_threshold": merge_affinity,
            "model_kind": affinity_checkpoint.get("model_kind", "pair_mlp"),
            "feature_mode": affinity_checkpoint.get("feature_mode", "full"),
            "training_appearance_descriptor_contract": affinity_checkpoint.get(
                "appearance_descriptor_contract", "unknown_legacy_checkpoint"
            ),
            "deployment_appearance_source": appearance_audit["source"],
            "deployment_siglip2_crop_summary_domain_validated": affinity_checkpoint.get(
                "deployment_siglip2_crop_summary_domain_validated", False
            ),
            "real_sam_fragment_noise_modeled": affinity_checkpoint.get(
                "real_sam_fragment_noise_modeled", False
            ),
            "feature_mode_scope": "learned_grouping; fixed_assignment_uses_association_configuration",
            "threshold_selection": affinity_checkpoint["threshold_selection"],
            "fragment_noise_contract": affinity_checkpoint.get(
                "fragment_noise_contract", {"fragment_noise_mode": "clean_oracle"}
            ),
            "training_sampling": affinity_checkpoint.get(
                "training_sampling", "pair_equal_legacy"
            ),
            "benchmark_labels_used_for_threshold_selection": False,
        }
    elif args.expected_learned_affinity_checkpoint_sha256:
        raise ValueError("unexpected learned affinity checkpoint SHA256")

    configuration = FragmentAssociationConfiguration(
        evidence_threshold=float(args.evidence_threshold),
        merge_affinity=merge_affinity,
        geometry_weight=float(args.geometry_weight),
        appearance_weight=float(args.appearance_weight),
        spatial_weight=float(args.spatial_weight),
        conflict_weight=float(args.conflict_weight),
        assignment_temperature=float(args.assignment_temperature),
        null_affinity=float(args.null_affinity),
        seed_policy=str(args.seed_policy),
        minimum_seed_support_views=int(args.minimum_seed_support_views),
        assignment_normalization=str(args.assignment_normalization),
        fragment_evidence_pooling=str(args.fragment_evidence_pooling),
        cannot_link_policy=str(args.cannot_link_policy),
    )
    result = build_fragment_object_hypotheses(
        fragment_positive=fragment_memory["fragment_positive"],
        element_centres=state["centres"],
        fragment_view_index=fragment_memory["fragment_view_index"],
        fragment_local_index=fragment_memory["fragment_local_index"],
        parent_index=fragment_memory["parent_index"],
        quality=fragment_memory["quality"],
        appearance_prototypes=appearance,
        view_visibility=fragment_memory["view_visibility"],
        learned_affinity_model=learned_affinity_model,
        configuration=configuration,
    )
    object_prototypes = (
        _select_hypothesis_prototypes(
            appearance,
            result["fragment_assignment"],
            fragment_memory["quality"],
            maximum_prototypes=int(args.maximum_object_prototypes),
            fragment_view_index=fragment_memory["fragment_view_index"],
        )
        if appearance is not None else {}
    )
    if "object_prototype_descriptors" in object_prototypes:
        object_prototypes["object_prototype_descriptors"] = object_prototypes[
            "object_prototype_descriptors"
        ].half()
    payload = validate_hypothesis_memory({
        "schema": SCHEMA,
        "scene_label": args.scene_label,
        "scene_state": str(state_path.resolve()),
        "scene_state_sha256": state["scene_state_sha256"],
        "fragment_memory": str(fragment_path.resolve()),
        "fragment_memory_sha256": fragment_memory["memory_sha256"],
        "fragment_count": int(result["fragment_assignment"].shape[0]),
        "surface_element_count": int(result["observed_membership"].shape[0]),
        "fragment_assignment": result["fragment_assignment"].half(),
        "fragment_null_probability": result["fragment_null_probability"].half(),
        "observed_membership": result["observed_membership"].half(),
        "hypothesis_seed_fragments": result["hypothesis_seed_fragments"],
        "hypothesis_representative_fragments": result[
            "hypothesis_representative_fragments"
        ],
        "association_configuration": asdict(configuration),
        "association_audit": result["pairwise_audit"],
        "learned_affinity_audit": learned_affinity_audit,
        "appearance_audit": appearance_audit,
        **object_prototypes,
        "representation": {
            "level": "global_object_hypothesis_memory",
            "historical_online_token_bootstrap_used": False,
            "global_pairwise_association": True,
            "pair_affinity_source": result["pairwise_audit"][
                "pair_affinity_source"
            ],
            "same_view_seed_cannot_link": configuration.cannot_link_policy != "disjoint_same_view",
            "fragment_evidence_pooling": configuration.fragment_evidence_pooling,
            "sam_hierarchy_containment_treated_as_object_identity": (
                configuration.seed_policy == "hierarchy_roots"
            ),
            "fragment_assignment": "top2_soft_plus_explicit_null",
            "membership_composition": configuration.fragment_evidence_pooling,
            "completion_performed": False,
            "prototype_selection": "distinct_source_views_first",
            "prototype_consensus": "distinct_source_views",
        },
        "information_policy": {
            "source_only": True,
            "query_free": True,
            "benchmark_labels_opened": False,
            "benchmark_masks_opened": False,
            "text_queries_opened": False,
            "target_rgb_opened": False,
        },
    })
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"object-hypothesis memory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, output)
    report = {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "scene_label": args.scene_label,
        **payload["association_audit"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-state", required=True)
    parser.add_argument("--expected-scene-state-sha256", required=True)
    parser.add_argument("--fragment-memory", required=True)
    parser.add_argument("--expected-fragment-memory-sha256", required=True)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument(
        "--appearance-source",
        choices=("none", "legacy_state_prototypes", "fragment_language_manifest"),
        default="none",
    )
    parser.add_argument("--fragment-language-manifest")
    parser.add_argument("--maximum-object-prototypes", type=int, default=8)
    parser.add_argument("--evidence-threshold", type=float, default=0.05)
    parser.add_argument("--merge-affinity", type=float, default=0.55)
    parser.add_argument("--geometry-weight", type=float, default=0.65)
    parser.add_argument("--appearance-weight", type=float, default=0.0)
    parser.add_argument("--spatial-weight", type=float, default=0.35)
    parser.add_argument("--conflict-weight", type=float, default=0.0)
    parser.add_argument("--learned-affinity-checkpoint")
    parser.add_argument("--expected-learned-affinity-checkpoint-sha256")
    parser.add_argument("--assignment-temperature", type=float, default=0.10)
    parser.add_argument("--null-affinity", type=float, default=0.45)
    parser.add_argument(
        "--seed-policy",
        choices=("all_fragments", "hierarchy_roots", "cross_view_supported"),
        default="all_fragments",
    )
    parser.add_argument("--minimum-seed-support-views", type=int, default=3)
    parser.add_argument(
        "--assignment-normalization",
        choices=("full_then_top2", "top2_then_normalize"),
        default="full_then_top2",
    )
    parser.add_argument(
        "--fragment-evidence-pooling",
        choices=("probabilistic_union", "same_view_max_then_union", "seed_medoid"),
        default="probabilistic_union",
    )
    parser.add_argument(
        "--cannot-link-policy",
        choices=(
            "exact_same_view",
            "disjoint_same_view",
            "exact_group_disjoint_assignment",
        ),
        default="exact_same_view",
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
