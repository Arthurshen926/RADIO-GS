"""Cold replay of fixed source-scale projection and native text-mask evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from radio_gs.data.lerf_dataset import _read_cameras_binary
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images
from radio_gs.v4.evaluation.lerf_common import camera, load_labels, load_text_cache, GENERIC_NEGATIVES
from radio_gs.v4.evaluation.lerf_object_ceiling import _load_state, _build_carrier
from radio_gs.v4.contracts.build_lerf_object_hypotheses import validate_hypothesis_memory
from radio_gs.v4.evaluation.lerf_object_hypothesis_text_evaluator import prototype_fragment_consensus_posterior, prototype_fragment_consensus_scores, cosine_consensus_set_posterior
from radio_gs.v4.evaluation.lerf_text_cold_evaluator import compose_object_queries, _load_fragment_prototypes
from radio_gs.scripts.evaluate_opengaussian_lerf_masks import FRAMES, score_masks
from radio_gs.v4.query.overlapping_candidates import compose_overlapping_candidates
from radio_gs.v4.evaluation.real_sam_token_association import _lift_masks
from radio_gs.v4.evaluation.lerf_source_mask_gate import _masks
from radio_gs.v4.object_memory.fragment_to_object import _compose_observed_membership
from radio_gs.v4.carrier import SurfaceVoxelCarrier
from radio_gs.v4.query.anchored_fragments import anchored_fragment_extent
from radio_gs.v4.contracts.lerf_fragment_surface_memory import validate_memory
from radio_gs.v4.contracts.surface_scene_bundle import SurfaceCarrierConfiguration
from radio_gs.v4.evaluation.selected_surface_render import render_selected_surface_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-report", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--positive-text-cache", type=Path)
    parser.add_argument("--negative-text-cache", type=Path)
    parser.add_argument("--compare-candidate-composition", action="store_true")
    parser.add_argument("--hypothesis-memory", type=Path)
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--compare-chain-stages", action="store_true")
    parser.add_argument("--compare-source-stages", action="store_true")
    parser.add_argument("--native-source-relift", action="store_true")
    parser.add_argument("--surface-replacement", type=Path)
    parser.add_argument("--anchored-source-only", action="store_true")
    parser.add_argument("--reuse-native-relift", type=Path)
    parser.add_argument("--pixel-threshold", type=float)
    parser.add_argument("--formal-fragment-memory", type=Path)
    parser.add_argument("--include-canonical", action="store_true")
    parser.add_argument("--selected-only-scene-render", action="store_true")
    args = parser.parse_args()
    if sum((args.compare_candidate_composition, args.compare_chain_stages, args.compare_source_stages)) > 1:
        parser.error("select only one comparison family")
    if args.reuse_native_relift and args.native_source_relift:
        parser.error("cannot reuse and rebuild native lifting together")
    if args.anchored_source_only and not args.compare_source_stages:
        parser.error("anchored-source-only requires compare-source-stages")
    if args.pixel_threshold is not None and not 0 < args.pixel_threshold < 1:
        parser.error("pixel threshold must lie in (0,1)")
    if args.formal_fragment_memory and (args.reuse_native_relift or args.native_source_relift or args.surface_replacement):
        parser.error("formal fragment memory cannot mix with another geometry/evidence override")
    torch.set_num_threads(4)
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.text_report.read_text())
    scene = reference["scene_label"]
    state = _load_state(Path(reference["scene_state"]), expected_sha256=reference["scene_state_sha256"])
    memory_path = args.hypothesis_memory or Path(reference["hypothesis_memory"])
    memory_hash = sha256_file(memory_path)
    if args.hypothesis_memory is None and memory_hash != reference["hypothesis_memory_sha256"]:
        raise ValueError("hypothesis hash mismatch")
    memory = validate_hypothesis_memory(torch.load(memory_path, map_location="cpu", weights_only=False))
    fragment_path = Path(memory["fragment_memory"])
    if sha256_file(fragment_path) != memory["fragment_memory_sha256"]:
        raise ValueError("fragment hash mismatch")
    fragment = torch.load(fragment_path, map_location="cpu", weights_only=False)
    if memory["scene_state_sha256"] != state["scene_state_sha256"] or fragment["scene_state_sha256"] != state["scene_state_sha256"]:
        raise ValueError("scene axes are not hash-bound")
    shape = tuple(fragment["raster_shape"])
    old = _build_carrier(state)
    state["method_configuration"]["carrier"]["reference_raster_shape"] = shape
    fixed = _build_carrier(state)
    sparse = args.scene_root / "sparse/0"
    views = _read_images(sparse / "images.bin", _read_cameras_binary(sparse / "cameras.bin"))
    for frame in fragment["source_frames"]:
        c = camera(views[frame], frame, *shape)
        first, second = old.project(c), fixed.project(c)
        for key in ("element_ids", "pixel_ids", "weights", "depths"):
            if not torch.equal(getattr(first, key), getattr(second, key)):
                raise ValueError(f"source projection changed: {frame} {key}")
    old._projection_cache.clear()
    fixed._projection_cache.clear()
    print(scene, "source replay exact", len(fragment["source_frames"]), flush=True)
    if args.formal_fragment_memory:
        formal = validate_memory(torch.load(args.formal_fragment_memory, weights_only=False, map_location="cpu"))
        formal_state = _load_state(Path(formal["scene_state"]), expected_sha256=formal["scene_state_sha256"])
        for key in ("centres", "normals", "confidence"):
            if not torch.equal(state[key], formal_state[key]):
                raise ValueError("formal fragment geometry differs")
        if SurfaceCarrierConfiguration.from_dict(formal_state["method_configuration"]["carrier"]) != SurfaceCarrierConfiguration.from_dict(state["method_configuration"]["carrier"]):
            raise ValueError("formal fragment carrier recipe differs")
        for key in ("fragment_frame_id", "fragment_view_index", "fragment_local_index", "quality", "parent_index"):
            if not torch.equal(fragment[key], formal[key]):
                raise ValueError("formal fragment identity axes differ")
        fragment = formal
        memory["observed_membership"] = _compose_observed_membership(fragment["fragment_positive"].float(), fragment["quality"], memory["fragment_assignment"], fragment["fragment_view_index"], pooling=memory["association_configuration"]["fragment_evidence_pooling"])
    if args.surface_replacement:
        if not args.native_source_relift or not args.compare_source_stages:
            raise ValueError("surface replacement is restricted to relifted source diagnostics")
        surface = torch.load(args.surface_replacement, weights_only=False, map_location="cpu")
        if surface["schema"] != "radio_gs.surface_object_memory_v4.calibrated_sparse_surface.v1" or surface["scene_label"] not in (scene, "lerf_" + scene):
            raise ValueError("replacement surface identity differs")
        # Native-only physical footprint, quantized in output pixels. No coarse
        # radius-one floor is magnified into a 16-pixel disk.
        fixed = SurfaceVoxelCarrier(surface["centres"], surface["voxel_size_colmap"],
            normals=surface["normals"], confidence=surface["confidence"],
            maximum_splat_radius=16, surface_band_voxels=1.5, maximum_contributors_per_pixel=8)
        state["method_configuration"]["carrier"] = {"voxel_size": surface["voxel_size_colmap"], "maximum_splat_radius": 16, "surface_band_voxels": 1.5, "maximum_contributors_per_pixel": 8, "reference_raster_shape": None}
    if args.reuse_native_relift:
        relift = torch.load(args.reuse_native_relift, weights_only=False, map_location="cpu")
        if relift["source_hypothesis_sha256"] != memory_hash or relift["source_fragment_sha256"] != memory["fragment_memory_sha256"]:
            raise ValueError("relift input hashes differ")
        fragment["fragment_positive"] = relift["fragment_positive"]
        fragment["view_visibility"] = relift["view_visibility"]
        memory["observed_membership"] = relift["observed_membership"]
    if args.native_source_relift:
        positives, visibility, receipts = [], [], []
        for row in fragment["frame_records"]:
            path = Path(row["sam_fragment_payload"])
            if sha256_file(path) != row["sam_fragment_payload_sha256"]:
                raise ValueError("source mask hash differs")
            payload = torch.load(path, weights_only=False, map_location="cpu")
            h, w = payload["mask_shape"]
            masks = _masks(path, h, w)
            positive, visible = _lift_masks(fixed, camera(views[row["frame_id"]], row["frame_id"], h, w), masks, torch.device("cuda:0"))
            positives.append(positive.half().cpu())
            visibility.append(visible[0].cpu())
            receipts.append({"frame": row["frame_id"], "raster": [h, w], "sha256": row["sam_fragment_payload_sha256"]})
            fixed._projection_cache.clear()
            print(scene, "native source relift", row["frame_id"], flush=True)
        fragment["fragment_positive"] = torch.cat(positives)
        fragment["view_visibility"] = torch.stack(visibility)
        memory["observed_membership"] = _compose_observed_membership(fragment["fragment_positive"].float(), fragment["quality"], memory["fragment_assignment"], fragment["fragment_view_index"], pooling=memory["association_configuration"]["fragment_evidence_pooling"])
        torch.save({"fragment_positive": fragment["fragment_positive"], "view_visibility": fragment["view_visibility"], "observed_membership": memory["observed_membership"], "receipts": receipts, "source_fragment_sha256": memory["fragment_memory_sha256"], "source_hypothesis_sha256": memory_hash, "groups_and_descriptors_frozen": True, "carrier_configuration": state["method_configuration"]["carrier"]}, out / "native_relift.pt")
    _, categories, _, _ = load_labels(args.label_root, scene)
    device = torch.device("cuda:0")
    text_path = args.positive_text_cache or Path(reference["text_query_cache"])
    negative_path = args.negative_text_cache or Path(reference["negative_text_query_cache"])
    text = load_text_cache(text_path, categories, device)
    negatives = load_text_cache(negative_path, list(GENERIC_NEGATIVES), device)
    config = reference["query_contract"]
    with torch.no_grad():
        scores = prototype_fragment_consensus_posterior(memory, text, negatives,
            temperature=config["temperature"], null_similarity=config["null_similarity"],
            consensus_fragments=config["distinct_fragment_consensus"])
        posterior, composition = compose_object_queries(memory["observed_membership"], scores.cpu(), maximum_query_tokens=None)
    torch.save({"element_probability": posterior, "categories": categories,
                "scene_state_sha256": reference["scene_state_sha256"], "hypothesis_memory_sha256": memory_hash,
                "diagnostic_derived_field_not_sealed_deployment": bool(args.native_source_relift or args.reuse_native_relift or args.surface_replacement or args.formal_fragment_memory),
                "replacement_surface_sha256": sha256_file(args.surface_replacement) if args.surface_replacement else None,
                "composition": composition}, out / "query_posterior.pt")
    threshold = reference["pixel_threshold"] if args.pixel_threshold is None else args.pixel_threshold
    alternative = None
    if args.compare_candidate_composition:
        alternative, alternative_contract = compose_overlapping_candidates(memory["observed_membership"], scores.cpu())
        torch.save({"element_support": alternative, "categories": categories, "contract": alternative_contract}, out / "candidate_support.pt")
    modes = (("mixture", fixed), ("overlap", fixed)) if args.compare_candidate_composition else (("fixed", fixed), ("old", old))
    if args.fixed_only:
        if args.compare_candidate_composition:
            raise ValueError("fixed-only cannot compare candidate composition")
        modes = (("fixed", fixed),)
    observations, frames = [], []
    fields = {mode: alternative if mode == "overlap" else posterior for mode, _ in modes}
    if args.compare_chain_stages:
        similarity = prototype_fragment_consensus_scores(memory, text, consensus_fragments=config["distinct_fragment_consensus"]).cpu()
        ids = similarity.argmax(-1)
        onehot = torch.zeros_like(similarity).scatter_(1, ids[:, None], 1.)
        mixed, _ = compose_object_queries(memory["observed_membership"], onehot)
        raw = memory["observed_membership"].float().cpu()
        fields = {
            "cosine_top1_raw": raw[:, ids],
            "generic_top1_raw": raw[:, scores.cpu().argmax(-1)],
            "cosine_top1_mixture": mixed,
            "cosine_set_raw": raw @ cosine_consensus_set_posterior(similarity, temperature=config["temperature"], set_mass=1., null_similarity=config["null_similarity"]).T,
        }
        modes = tuple((mode, fixed) for mode in fields)
        torch.save({"fields": fields, "categories": categories, "similarity": similarity, "generic_scores": scores.cpu(), "selected_ids": ids, "hypothesis_memory_sha256": memory_hash}, out / "chain_fields.pt")
    if args.compare_source_stages:
        flat, _, _ = _load_fragment_prototypes(Path(memory["appearance_audit"]["fragment_language_manifest"]), state=state)
        count = fragment["fragment_positive"].shape[0]
        if flat.shape[0] != 2 * count:
            raise ValueError("fragment descriptor axes do not align")
        sim = torch.nn.functional.normalize(text.cpu(), dim=-1) @ torch.nn.functional.normalize(flat, dim=-1).T
        masked_id = sim[:, :count].argmax(-1)
        dual_id = torch.maximum(sim[:, :count], sim[:, count:]).argmax(-1)
        fields = {
            "source_masked_top1": fragment["fragment_positive"][masked_id].float().T,
            "source_dual_top1": fragment["fragment_positive"][dual_id].float().T,
        }
        if args.anchored_source_only:
            anchored, anchor_audit = anchored_fragment_extent(fragment["fragment_positive"], sim[:, :count], fragment["fragment_view_index"])
            fields = {"source_anchored_extent": anchored}
            torch.save(anchor_audit, out / "anchor_audit.pt")
        if args.include_canonical:
            fields["canonical_full"] = posterior
        modes = tuple((mode, fixed) for mode in fields)
        torch.save({"fields": fields, "categories": categories, "masked_ids": masked_id, "dual_ids": dual_id, "similarity": sim, "fragment_memory_sha256": memory["fragment_memory_sha256"]}, out / "source_fields.pt")
    for mode, carrier in modes:
        query_field = fields[mode]
        folder = out / mode
        folder.mkdir()
        for frame in FRAMES[scene]:
            targets = sorted((args.gt_root / f"frame_{frame:05d}").glob("*.jpg"))
            if not targets:
                raise FileNotFoundError("missing public GT frame")
            with Image.open(targets[0]) as image:
                width, height = image.size
            cam = camera(views[frame], frame, height, width)
            support = carrier.render_posterior(torch.ones(carrier.num_elements), cam).numpy() > 0
            frames.append({"mode": mode, "frame": frame, "coverage": float(support.mean())})
            for target in targets:
                if target.stem not in categories:
                    raise ValueError("public query inventory differs from cached text")
                index = categories.index(target.stem)
                if args.selected_only_scene_render:
                    mask = render_selected_surface_support(carrier, query_field[:, index], cam, threshold=threshold).numpy()
                else:
                    mask = carrier.render_posterior(query_field[:, index], cam).numpy() >= threshold
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"frame_{frame:05d}_{target.stem}.png")
                with Image.open(target) as image:
                    gt = np.asarray(image) > 10
                observations.append({"mode": mode, "frame": frame, "object": target.stem,
                    "support_upper_bound": float((support & gt).sum() / gt.sum()) if gt.any() else 0.0})
            carrier._projection_cache.clear()
            print(scene, mode, frame, "done", flush=True)
    result = {"scene": scene, "development_only": True, "source_projection_exact": True,
        "source_frame_count": len(fragment["source_frames"]), "carrier_configuration": state["method_configuration"]["carrier"],
        "source_state_sha256": reference["scene_state_sha256"], "hypothesis_memory": str(memory_path.resolve()), "hypothesis_memory_sha256": memory_hash,
        "text_query_cache": str(text_path.resolve()), "text_query_cache_sha256": sha256_file(text_path),
        "negative_text_query_cache": str(negative_path.resolve()), "negative_text_query_cache_sha256": sha256_file(negative_path),
        "query_posterior_sha256": sha256_file(out / "query_posterior.pt"), "pixel_threshold": threshold,
        "reference_pixel_threshold": reference["pixel_threshold"], "explicit_threshold_override": args.pixel_threshold is not None,
        "render_semantics": "selected_only_binary_surfel_support" if args.selected_only_scene_render else "full_scene_probability_then_pixel_threshold",
        "threshold_tuned": False, "membership_changed": args.hypothesis_memory is not None, "native_frame_coverage": frames,
        "support_observations": observations, "scores": {mode: score_masks(args.gt_root, out / mode, scene) for mode, _ in modes}}
    if args.compare_candidate_composition:
        result["candidate_contract"] = alternative_contract
        result["candidate_support_sha256"] = sha256_file(out / "candidate_support.pt")
    if args.compare_chain_stages:
        result["chain_contract"] = {"top1_is_single_target_diagnostic": True, "cosine_set_raw_is_query_conditioned_convex_mixture": True, "generic_negative_activation_mean": float((scores >= threshold).float().mean()), "chain_fields_sha256": sha256_file(out / "chain_fields.pt")}
    if args.compare_source_stages:
        result["source_contract"] = {"top1_is_single_fragment_diagnostic": True, "object_aggregation_bypassed": True, "source_fields_sha256": sha256_file(out / "source_fields.pt")}
    if args.native_source_relift:
        result["membership_changed"] = True
        result["native_relift_contract"] = {"groups_and_descriptors_frozen": True, "source_only": True, "payload_sha256": sha256_file(out / "native_relift.pt"), "not_a_sealed_deployment_bundle": True}
    if args.reuse_native_relift:
        result["membership_changed"] = True
        result["reused_native_relift_sha256"] = sha256_file(args.reuse_native_relift)
    if args.formal_fragment_memory:
        result["membership_changed"] = True
        result["formal_fragment_memory"] = str(args.formal_fragment_memory.resolve())
        result["formal_fragment_memory_sha256"] = sha256_file(args.formal_fragment_memory)
    if args.surface_replacement:
        result["source_projection_exact"] = False
        result["replacement_geometry"] = {"path": str(args.surface_replacement.resolve()), "sha256": sha256_file(args.surface_replacement), "element_count": fixed.num_elements, "old_state_is_descriptor_inventory_only": True}
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
