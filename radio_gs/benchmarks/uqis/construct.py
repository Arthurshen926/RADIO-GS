"""Build geometry-verified query candidates from fixed official instances.

Only the constructor sees mesh instance labels and RGB-D. Query construction
does not load method features, predictions, thresholds, or benchmark scores.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, laplace
from scipy.spatial import cKDTree

from . import VERSION
from .sensor import SensorIndex


@dataclass(frozen=True)
class ConstructionConfig:
    candidate_frame_stride: int = 20
    correspondence_m: float = .05
    extent_correspondence_m: float = .08
    minimum_visible_depth_pixels: int = 128
    minimum_interior_px: float = 3.
    minimum_image_border_px: int = 8
    minimum_rgb_border_px: int = 8
    crop_padding: float = .15
    crop_size: int = 224
    minimum_mapping_coverage: float = .70
    minimum_mapping_views: int = 5


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def projected_observation(sensor, index, tree, mesh_ids):
    pose, depth, rgb = sensor.read(index)
    yy, xx = np.mgrid[:sensor.height, :sensor.width]
    k = sensor.k_depth
    valid = np.isfinite(depth) & (depth > .2) & (depth < 8.)
    camera = np.stack(((xx-k[0, 2])*depth/k[0, 0], (yy-k[1, 2])*depth/k[1, 1], depth), axis=-1)
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    dist = np.full(depth.shape, np.inf)
    nearest = np.zeros(depth.shape, dtype=np.int64)
    dist[valid], nearest[valid] = tree.query(world[valid])
    return pose, depth, rgb, world, dist, nearest, mesh_ids[nearest]


def registered_crop(rgb, mask, kd, kc, config):
    """Square natural RGB crop, located using camera intrinsics, not resize."""
    v, u = np.nonzero(mask)
    color_u = kc[0, 0] * (u-kd[0, 2]) / kd[0, 0] + kc[0, 2]
    color_v = kc[1, 1] * (v-kd[1, 2]) / kd[1, 1] + kc[1, 2]
    cx, cy = .5*(color_u.min()+color_u.max()), .5*(color_v.min()+color_v.max())
    side = int(np.ceil(max(np.ptp(color_u)+1, np.ptp(color_v)+1)*(1+2*config.crop_padding)))
    x0, y0 = int(np.floor(cx-side/2)), int(np.floor(cy-side/2))
    box = (x0, y0, x0+side, y0+side)
    crop = rgb.crop(box).resize((config.crop_size, config.crop_size), Image.Resampling.BICUBIC)
    y, x = np.mgrid[:config.crop_size, :config.crop_size]
    uc, vc = x0+(x+.5)*side/config.crop_size, y0+(y+.5)*side/config.crop_size
    ud = np.rint(kd[0, 0]*(uc-kc[0, 2])/kc[0, 0]+kd[0, 2]).astype(int)
    vd = np.rint(kd[1, 1]*(vc-kc[1, 2])/kc[1, 1]+kd[1, 2]).astype(int)
    inside = (ud>=0)&(vd>=0)&(ud<mask.shape[1])&(vd<mask.shape[0])
    foreground = np.zeros(inside.shape, bool)
    foreground[inside] = mask[vd[inside], ud[inside]]
    gray = np.asarray(crop.convert("L"), dtype=np.float32)/255.
    interior = binary_erosion(foreground, iterations=1)
    sharpness = float(np.mean(laplace(gray)[interior]**2)) if interior.any() else 0.
    exposure = float(np.mean((gray[foreground]>.03)&(gray[foreground]<.97))) if foreground.any() else 0.
    quality = {"foreground_fraction":float(foreground.mean()), "foreground_laplacian_energy":sharpness,
               "usable_exposure_fraction":exposure, "square_rgb_box":list(box)}
    return crop, quality


def choose_prompt(extent, reliable, world, config):
    # The exterior of the image is explicitly background for the distance map.
    interior = distance_transform_edt(np.pad(extent, 1))[1:-1, 1:-1]
    legal = reliable & (interior >= config.minimum_interior_px)
    margin = config.minimum_image_border_px
    legal[:margin] = legal[-margin:] = False
    legal[:, :margin] = legal[:, -margin:] = False
    if not legal.any():
        return None
    rank = np.where(legal, interior, -1.)
    v, u = np.unravel_index(np.argmax(rank), rank.shape)
    return {"positive_pixel_uv":[int(u), int(v)], "point_world_xyz":world[v, u].tolist(),
            "interior_distance_px":float(interior[v, u]),
            "image_border_px":int(min(u, v, extent.shape[1]-1-u, extent.shape[0]-1-v))}


def select_mapping_supported_queries(rows, targets, sensor, visible_by_frame, config):
    """Choose references jointly: a query view cannot also be mapping evidence.

    Start from each target's best image. If withholding those cameras removes
    too much mapping evidence, choose the single reference change that reduces
    the visibility deficit most, then preserves the most image quality. This
    uses construction geometry only; an unresolved scene stays a candidate.
    """
    selected = {t['instance_id']:max((r for r in rows if r['instance_id']==t['instance_id']),
                                    key=lambda r:r['score']) for t in targets}
    frame_ids = sorted(visible_by_frame)
    instances = sorted(selected)
    visibility = np.array([[i in visible_by_frame[f] for i in instances] for f in frame_ids])
    exclusions = {f:np.isin(frame_ids, sensor.exclude([f])[1])
                  for f in sorted({r['frame_index'] for r in rows})}

    def counts(choices):
        legal = ~np.logical_or.reduce([exclusions[r['frame_index']] for r in choices.values()])
        return visibility[legal].sum(axis=0)

    def deficit(count):
        return int(np.maximum(config.minimum_mapping_views-count, 0).sum())

    count = counts(selected)
    while deficit(count):
        proposal, proposal_count, best_rank = None, None, None
        for row in rows:
            alternative = {**selected, row['instance_id']:row}
            next_count = counts(alternative)
            if deficit(next_count) >= deficit(count):
                continue
            rank = (deficit(next_count), -sum(r['score'] for r in alternative.values()))
            if best_rank is None or rank < best_rank:
                proposal, proposal_count, best_rank = alternative, next_count, rank
        if proposal is None:
            break
        selected, count = proposal, proposal_count
    return selected, {str(i):int(n) for i,n in zip(instances,count)}


def build_scene(job):
    scene, targets, source, domain, output, config_data = job
    config = ConstructionConfig(**config_data)
    output = Path(output)/"construction"/scene
    output.mkdir(parents=True, exist_ok=True)
    config_id = hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()
    if (output/"scene.json").exists():
        cached = json.loads((output/"scene.json").read_text())
        if cached['construction_config_sha256'] != config_id:
            raise ValueError("Choose a new output directory for another construction policy")
        return cached
    sensor = SensorIndex(source['sens']['path'])
    xyz = np.load(domain['mesh_xyz_path'])
    ids = np.load(domain['mesh_instance_ids_path'])
    tree = cKDTree(xyz)
    best = {}
    rows = []
    coverage_samples = {}
    visible_by_frame = {}
    nviews = 0
    yy, xx = np.mgrid[:sensor.height, :sensor.width]
    kd, kc = sensor.k_depth, sensor.k_color
    rgb_u = kc[0,0]*(xx-kd[0,2])/kd[0,0]+kc[0,2]
    rgb_v = kc[1,1]*(yy-kd[1,2])/kd[1,1]+kc[1,2]
    rgb_visible = (rgb_u>=0)&(rgb_v>=0)&(rgb_u<sensor.color_width)&(rgb_v<sensor.color_height)
    rgb_margin = config.minimum_rgb_border_px
    rgb_interior = (rgb_u>=rgb_margin)&(rgb_v>=rgb_margin)&(rgb_u<sensor.color_width-rgb_margin)&(rgb_v<sensor.color_height-rgb_margin)
    for index in range(0, len(sensor.frames), config.candidate_frame_stride):
        if not np.isfinite(sensor.frames[index][0]).all():
            continue
        pose, depth, rgb, world, distance, nearest, labels = projected_observation(sensor, index, tree, ids)
        nviews += 1
        # Actual sampled depth surfaces, retained only for constructor coverage.
        stride_mask = np.zeros(depth.shape, bool)
        stride_mask[::4, ::4] = True
        coverage_samples[index] = world[stride_mask & (depth>.2) & (depth<8.)].astype(np.float32)
        visible_by_frame[index] = set(map(int, np.unique(labels[stride_mask & (distance<=config.extent_correspondence_m)])))
        for target in targets:
            instance = target['instance_id']
            extent = (labels==instance)&(distance<=config.extent_correspondence_m)&rgb_visible
            if int(extent.sum()) < config.minimum_visible_depth_pixels:
                continue
            reliable = extent & (distance<=config.correspondence_m) & rgb_interior
            prompt = choose_prompt(extent, reliable, world, config)
            if prompt is None:
                continue
            crop, quality = registered_crop(rgb, extent, sensor.k_depth, sensor.k_color, config)
            # Bound each component so one noisy/sharp patch cannot win solely
            # by high-frequency energy. This is a construction rule, not fitted.
            sharp = quality['foreground_laplacian_energy']/(quality['foreground_laplacian_energy']+.0025)
            area = min(1., float(extent.sum())/(.10*extent.size))
            exposure = .1+.9*quality['usable_exposure_fraction']
            occupancy = .1+.9*quality['foreground_fraction']
            score = float((.1+.9*sharp)*np.sqrt(area)*exposure*np.sqrt(occupancy))
            u, v = prompt['positive_pixel_uv']
            row = {"instance_id":instance, "frame_index":index, "score":score,
                "visible_depth_pixels":int(extent.sum()), "reliable_depth_pixels":int(reliable.sum()),
                "prompt_distance_m":float(distance[v,u]), "camera_to_world":pose.tolist(),
                "camera_intrinsics":sensor.k_depth[:3,:3].tolist(),
                "raster_size":[sensor.width, sensor.height], "click_depth_m":float(depth[v,u]),
                **prompt, "image_quality":quality}
            rows.append({k:row[k] for k in ['instance_id','frame_index','score','visible_depth_pixels','prompt_distance_m','image_quality']})
            if instance not in best or score > best[instance]['score']:
                crop_path = output/f"instance_{instance:03d}.png"
                crop.save(crop_path)
                best[instance] = {**row, "crop_rgb_path":str(crop_path.resolve())}
        if nviews % 40 == 0:
            print(scene, 'frames', nviews, 'targets', len(best), '/', len(targets), flush=True)
    selected, visibility_counts = select_mapping_supported_queries(
        rows, [t for t in targets if t['instance_id'] in best], sensor, visible_by_frame, config)
    reference_changes = []
    for instance, selected_row in selected.items():
        index = selected_row['frame_index']
        if index == best[instance]['frame_index']:
            continue
        reference_changes.append({'instance_id':instance,'previous_frame':best[instance]['frame_index'],
                                  'selected_frame':index,'reason':'preserve_mapping_visibility'})
        pose, depth, rgb, world, distance, nearest, labels = projected_observation(sensor,index,tree,ids)
        extent = (labels==instance)&(distance<=config.extent_correspondence_m)&rgb_visible
        reliable = extent&(distance<=config.correspondence_m)&rgb_interior
        prompt = choose_prompt(extent,reliable,world,config)
        crop, quality = registered_crop(rgb,extent,sensor.k_depth,sensor.k_color,config)
        crop_path = output/f'instance_{instance:03d}.png'
        crop.save(crop_path)
        u,v = prompt['positive_pixel_uv']
        best[instance] = {**selected_row,**prompt,'camera_to_world':pose.tolist(),
            'camera_intrinsics':sensor.k_depth[:3,:3].tolist(),'raster_size':[sensor.width,sensor.height],
            'click_depth_m':float(depth[v,u]),'reliable_depth_pixels':int(reliable.sum()),
            'image_quality':quality,'crop_rgb_path':str(crop_path.resolve())}
    query_indices = sorted({r['frame_index'] for r in best.values()})
    legal, withheld = sensor.exclude(query_indices)
    legal_set = set(map(int, legal))
    surfaces = np.concatenate([points for i,points in coverage_samples.items() if i in legal_set])
    mapping_tree = cKDTree(surfaces)
    coverage = {}
    for target in targets:
        distance, _ = mapping_tree.query(xyz[ids==target['instance_id']])
        coverage[str(target['instance_id'])] = float(np.mean(distance<=config.correspondence_m))
    missing = [t['instance_id'] for t in targets if t['instance_id'] not in best]
    low_coverage = [int(k) for k,v in coverage.items() if v<config.minimum_mapping_coverage]
    low_visibility = [int(k) for k,v in visibility_counts.items() if v<config.minimum_mapping_views]
    result = {"version":VERSION, "scene_id":scene, "construction_config_sha256":config_id,
        "input_sens":source['sens'], "scanned_query_candidate_views":nviews,
        "requested_target_count":len(targets), "selected_target_count":len(best),
        "missing_target_instances":missing, "low_mapping_coverage_instances":low_coverage,
        "low_mapping_visibility_instances":low_visibility,
        "mapping_visible_sampled_frames":visibility_counts,
        "reference_changes_for_mapping":reference_changes,
        "query_frame_ids":[f'{i:06d}' for i in query_indices],
        "field_frame_ids":[f'{i:06d}' for i in legal],
        "withheld_frame_ids":[f'{i:06d}' for i in withheld],
        "mapping_surface_coverage":coverage,
        "targets":[best[k] for k in sorted(best)],
        "status":"constructed" if not missing and not low_coverage and not low_visibility else "needs_construction_review"}
    write_json(output/'candidates.json', rows)
    write_json(output/'scene.json', result)
    print(scene, result['status'], len(best), 'of', len(targets), 'targets', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--text-profile', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    config = ConstructionConfig(**json.loads(args.config.read_text()))
    profile = json.loads(args.text_profile.read_text())
    evaluator = json.loads((args.legacy_root/'construction_frozen_candidate_v1/target_manifest.evaluator.json').read_text())
    receipts = json.loads((args.legacy_root/'official_construction_v5/construction_receipts.json').read_text())
    jobs = []
    for row in receipts['scenes']:
        scene = row['scene_id']
        jobs.append((scene, [t for t in profile['targets'] if t['scene_id']==scene], row['sources'],
            next(d for d in evaluator['scene_domains'] if d['scene_id']==scene), str(args.output), asdict(config)))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'construction_config.json', asdict(config))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(build_scene, jobs))
    write_json(args.output/'construction_summary.json', {"version":VERSION,
        "scene_count":len(results), "requested_target_count":len(profile['targets']),
        "constructed_target_count":sum(r['selected_target_count'] for r in results),
        "complete":all(r['status']=='constructed' for r in results),
        "scenes":[{k:r[k] for k in ['scene_id','status','missing_target_instances','low_mapping_coverage_instances','low_mapping_visibility_instances','scanned_query_candidate_views','selected_target_count']} for r in results]})


if __name__ == '__main__':
    main()
