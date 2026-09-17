"""Source-only causal pilot: camera-clearance before vs depth weights after rasterization."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.camera_clearance import camera_plane_clearance_confidence
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions, marginal_responsibility_statistics
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.utils.immutable_artifacts import sha256_file

p = argparse.ArgumentParser()
p.add_argument('--fusion-report', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
torch.set_num_threads(4)
prior = MogeGeometryPrior(a.fusion_report, sha256_file(a.fusion_report))
source = next(r for r in prior.report['geometry_receipt']['inputs'] if r['role']=='source_authority')
authority_path = Path(source['path'])
authority = json.loads(authority_path.read_text())
meta = authority['metadata']
model, _, renderer, _, _, _, _ = load_render_pipeline(meta['config'], meta['checkpoint'], torch.device('cuda:0'), strict_checkpoint_contract=True, load_ply_rgb_features=False)
frame = authority['frame_indices'][0]
camera = prior.cameras[frame]
pose = camera.world_to_camera.cuda()
h, w = meta['feature_height'], meta['feature_width']
clearance = camera_plane_clearance_confidence(model.get_xyz(), model.get_rotation(), model.get_scaling(), pose, near_plane=renderer.near_plane)
# Explicit diagnostic: remove source-depth free-space centres before visibility.
# This per-view gate is not persisted as a scene-wide deletion policy.
record = prior.records[frame]
prediction_path = record['prediction']
assert sha256_file(prediction_path) == record['prediction_sha256']
prediction = torch.load(prediction_path, map_location='cpu', weights_only=False)
calibration = prior.calibration[frame]
assert calibration['accepted']
depth = prediction['point_map'][..., 2].float().cuda() * prior.scale * calibration['scale'] + calibration['offset']
valid = prediction['validity'].bool().cuda()
xyz_camera = model.get_xyz() @ pose[:3, :3].T + pose[:3, 3]
k = camera.intrinsic.cuda().clone()
k[0] *= depth.shape[1] / camera.width
k[1] *= depth.shape[0] / camera.height
projected = xyz_camera @ k.T
uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
x, y = uv[:, 0].floor().long(), uv[:, 1].floor().long()
inside = (xyz_camera[:, 2] > 0) & (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
x, y = x.clamp(0, depth.shape[1]-1), y.clamp(0, depth.shape[0]-1)
d = depth[y, x]
known = inside & valid[y, x] & torch.isfinite(d) & (d > 0)
free_space = known & (xyz_camera[:, 2] < d - (0.08 * prior.scale + .02*d))
moge_opacity = clearance.confidence * (~free_space).float()
results = {}
for name, scale in [('original', None), ('camera_clearance_before_compositor', clearance.confidence),
                    ('source_depth_free_space_before_compositor', moge_opacity)]:
    hits = rasterize_single_view_contributions(model, renderer, pose, height=h, width=w, opacity_scale=scale)
    weights = marginal_responsibility_statistics(hits['pixel_ids'], hits['weights'], num_pixels=h*w).target_weight
    prior.apply(frame_index=frame, xyz=model.get_xyz().detach().cpu(), world_to_camera=pose,
                intrinsic=renderer.scaled_intrinsics(w,h), gaussian_ids=hits['gaussian_ids'],
                pixel_ids=hits['pixel_ids'], weights=weights, height=h, width=w)
    results[name] = prior.statistics[-1]
    results[name]['covered_pixels'] = int(hits['pixel_ids'].unique().numel())
    results[name]['weighted_depth_compatibility'] = 2*float(prior.statistics[-1]['conditioned_mass']) / max(float(weights.sum()), 1e-12)-1
report = {'role':'single_source_view_geometry_diagnostic_not_task_metric', 'frame_index':frame,
          'source_authority_sha256':source['sha256'], 'calibration_report_sha256':sha256_file(a.fusion_report),
          'checkpoint_sha256':sha256_file(meta['checkpoint']), 'query_or_label_access':False,
          'note':'single-view source depth free-space culling is diagnostic, not a promoted global geometry update; no geometry checkpoint is modified',
          'results':results}
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
