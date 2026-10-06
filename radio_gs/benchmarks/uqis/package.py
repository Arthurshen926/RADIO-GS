"""Export the repaired candidate as separate method, mapping and evaluator data."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import shutil

import numpy as np

from . import MODALITIES, VERSION
from .construct import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(legacy_root, profile_path, review_path, construction_root, output, visual_review_path=None):
    read = lambda p: json.loads(Path(p).read_text())
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Candidate packages are immutable: {output}")
    summary = read(Path(construction_root)/'construction_summary.json')
    if not summary['complete']:
        raise ValueError('All original targets and post-exclusion coverage must be resolved before export')
    old = read(Path(legacy_root)/'construction_frozen_candidate_v1/target_manifest.evaluator.json')
    profile = read(profile_path)
    review = read(review_path)
    review_by_target = {(t['scene_id'],t['instance_id']):t for t in review['targets']}
    text_by_target = {(t['scene_id'],t['instance_id']):t for t in profile['targets']}
    source_receipts = read(Path(legacy_root)/'official_construction_v5/construction_receipts.json')
    source_by_scene = {s['scene_id']:s['sources'] for s in source_receipts['scenes']}
    public = output/'method'
    private = output/'evaluator'
    queries = {m:[] for m in MODALITIES}
    domains, private_domains, targets, mapping = [], [], [], []
    salt = secrets.token_bytes(32)
    for domain in old['scene_domains']:
        scene = domain['scene_id']
        constructed = read(Path(construction_root)/'construction'/scene/'scene.json')
        selected = {r['instance_id']:r for r in constructed['targets']}
        geometry = public/'scenes'/scene/'mesh_xyz.npy'
        labels = private/'scenes'/scene/'mesh_instance_ids.npy'
        geometry.parent.mkdir(parents=True, exist_ok=True)
        labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(domain['mesh_xyz_path'], geometry)
        shutil.copyfile(domain['mesh_instance_ids_path'], labels)
        d = {"scene_id":scene, "mesh_vertices":domain['mesh_vertices'],
             "mesh_xyz_path":str(geometry.relative_to(public)), "mesh_xyz_sha256":digest(geometry)}
        domains.append(d)
        private_domains.append({**d,"mesh_instance_ids_path":str(labels.relative_to(private)),
                                "mesh_instance_ids_sha256":digest(labels)})
        aggregation = read(source_by_scene[scene]['aggregation']['path'])
        raw_labels = {int(g['objectId'])+1:g['label'] for g in aggregation['segGroups']}
        present_instances = set(map(int, np.unique(np.load(domain['mesh_instance_ids_path']))))
        for old_target in old['targets']:
            if old_target['scene_id'] != scene:
                continue
            instance = old_target['instance_id']
            selected_query = selected[instance]
            text = text_by_target[(scene,instance)]['expression']
            tags = review_by_target[(scene,instance)]
            if tags['expression_sha256'] != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError('Language review does not describe the bound expression')
            rows = {
                "text":{"expression":text},
                "image":{"crop_rgb_sha256":digest(selected_query['crop_rgb_path'])},
                "point_2d":{k:selected_query[k] for k in ['camera_to_world','camera_intrinsics','raster_size','positive_pixel_uv']},
                "point_3d":{"point_world_xyz":selected_query['point_world_xyz']},
            }
            pairing = {}
            for modality, row in rows.items():
                body = {"benchmark_version":VERSION,"scene_id":scene,"modality":modality,**row}
                qid = 'uq_'+hmac.new(salt,json.dumps(body,sort_keys=True,separators=(',',':')).encode(),hashlib.sha256).hexdigest()[:32]
                pairing[modality] = qid
                if modality=='image':
                    asset = public/'assets'/f'{qid}.png'
                    asset.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(selected_query['crop_rgb_path'],asset)
                    body['crop_rgb_path'] = str(asset.relative_to(public))
                queries[modality].append({"query_id":qid,**body})
            targets.append({"scene_id":scene,"instance_id":instance,
                "raw_semantic_label":old_target['raw_semantic_label'],"target_scope":"official_annotated_instance",
                "text_annotation_source":text_by_target[(scene,instance)]['expression_source'],
                "text_annotation_id":text_by_target[(scene,instance)]['expression_annotation_id'],
                "queries":pairing,"language_tags":tags['language_tags'],
                "language_review_source":tags['review_source'],
                "independent_identity_validation":False,
                "same_class_distractor_instance_ids":[i for i,label in raw_labels.items() if i!=instance and i in present_instances and label==raw_labels[instance]],
                "query_frame_id":f"{selected_query['frame_index']:06d}",
                "query_construction":selected_query,
                "mapping_surface_coverage":constructed['mapping_surface_coverage'][str(instance)],
                "mapping_visible_sampled_frames":constructed['mapping_visible_sampled_frames'][str(instance)]})
        mapping.append({"scene_id":scene,"sens_path":source_by_scene[scene]['sens']['path'],
            "field_frame_ids":constructed['field_frame_ids'],"withheld_frame_ids":constructed['withheld_frame_ids'],
            "query_frame_ids":constructed['query_frame_ids'],"source_permissions":"mapping_only"})
    for modality, rows in queries.items():
        write_json(public/'queries'/f'{modality}.json',{"benchmark_version":VERSION,"modality":modality,"queries":sorted(rows,key=lambda q:q['query_id'])})
    write_json(public/'scenes.json',{"benchmark_version":VERSION,"scene_domains":domains})
    write_json(private/'targets.json',{"benchmark_version":VERSION,"scene_domains":private_domains,"targets":targets})
    write_json(private/'query_id_key.json',{"hmac_key_hex":salt.hex()})
    if visual_review_path is not None:
        visual = read(visual_review_path)
        selected_by_target = {(t['scene_id'],t['instance_id']):t['query_construction'] for t in targets}
        if len(visual['targets']) != len(targets):
            raise ValueError('The visual review must cover every constructed reference')
        for row in visual['targets']:
            bound = selected_by_target[(row['scene_id'],row['instance_id'])]
            if row['crop_sha256'] != digest(bound['crop_rgb_path']):
                raise ValueError('Visual review refers to a different reference crop')
        write_json(private/'reference_review.json',visual)
    write_json(output/'mapping'/'scenes.json',{"benchmark_version":VERSION,"scenes":mapping})
    write_json(output/'release.json',{"benchmark_version":VERSION,"role":"opened_development_candidate",
        "scene_count":len(domains),"target_count":len(targets),"query_count":sum(map(len,queries.values())),
        "query_access":"map_only","point_2d_rgb_allowed":False,
        "primary_cohort":"all_original_67_targets","primary_metrics":"per_modality_scene_macro_AP_and_equal_modality_mean",
        "fixed_iou":"diagnostic_until_independent_development_calibration",
        "text_groups":"overlapping_literal_language_tags_not_certified_relation_free",
        "reference_quality":"assistant_visual_review_completed_with_unresolved_cases" if visual_review_path else "automatic_selection_with_assistant_visual_review_pending",
        "source_geometry":"registered_ScanNet_RGBD_and_official_mesh_for_construction",
        "geometry_track":"official_mesh_RGBD_assisted",
        "geometry_holdout":"reference_RGBD_observations_excluded_but_official_mesh_is_full_scan_geometry",
        "query_raster":"native_depth_camera_pixel_coordinates_no_depth_delivered_to_method",
        "crop_geometry":"square_RGB_crop_registered_with_color_and_depth_intrinsics",
        "construction_config":read(Path(construction_root)/'construction_config.json'),
        "old_prediction_reuse_allowed":False,"maps_must_use_this_versions_field_frame_ids":True})
    return {"package":str(output),"scenes":len(domains),"targets":len(targets),"queries":sum(map(len,queries.values()))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--text-profile',type=Path,required=True)
    p.add_argument('--text-review',type=Path,required=True)
    p.add_argument('--construction',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--visual-review',type=Path)
    a=p.parse_args()
    print(json.dumps(package(a.legacy_root,a.text_profile,a.text_review,a.construction,a.output,a.visual_review),indent=2))


if __name__=='__main__':
    main()
