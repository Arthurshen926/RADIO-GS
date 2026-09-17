"""Source-heldout full-dimensional RADIO reconstruction on sealed geometry.

Diagnostic uncompressed least-squares field, not a promoted compact semantic
model. Uses fresh exact contributions; never reads historical Gaussian rows.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.utils.immutable_artifacts import sha256_file

p = argparse.ArgumentParser()
p.add_argument('--scene', required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
torch.set_num_threads(4)
device = 'cuda:0'
report_path = a.root / a.scene / 'report.json'
report = json.loads(report_path.read_text())
g = report['results']['independent_scale']
model = FrozenGaussianGeometry.load(g['geometry'], expected_sha256=g['geometry_sha256'], device=device)
binding = model.fingerprint()
prior = MogeGeometryPrior(report['source_report'], report['source_report_sha256'])
feature_dir = Path('/mnt/pool/sqy/results/RADIO-GS/output/canonical_teacher_features') / a.scene
manifest_path = feature_dir / 'frame_manifest.json'
manifest = json.loads(manifest_path.read_text())
records = {r['frame']['frame_idx']: r for r in manifest['output_bundle']['frames']}
heldout = report['heldout_frames'][0]
# Deterministic temporal neighbors, selected without inspecting feature values.
training = sorted(report['mapping_frames'], key=lambda f: (abs(f-heldout), f))[:4]
assert heldout not in training
matrices, targets, receipts, alpha_maps = [], [], [], []
for frame in training + [heldout]:
    rec = next(t for t in records[frame]['tensors'] if t['relative_path'].startswith('backbone/'))
    path = feature_dir / rec['relative_path']
    assert sha256_file(path) == rec['sha256']
    feature = torch.load(path, map_location='cpu', weights_only=True)
    assert list(feature.shape) == rec['shape'] and torch.isfinite(feature).all()
    d, h, w = feature.shape
    target = F.normalize(feature.float().reshape(d, -1).T, dim=1)
    camera = prior.cameras[frame]
    k = camera.intrinsic.clone()
    k[0] *= w/camera.width
    k[1] *= h/camera.height
    # Match the geometry evaluator's clipping planes in calibrated world units.
    unit = prior.scale
    renderer = FeatureFieldRenderer(h, w, float(k[0,0]), float(k[1,1]),
        float(k[0,2]), float(k[1,2]), near_plane=.001*unit, far_plane=100*unit).to(device)
    pose = camera.world_to_camera.float().to(device)
    hits = rasterize_single_view_contributions(model, renderer, pose, height=h, width=w)
    alpha = hits['accumulated_alpha'].reshape(-1)
    weights = hits['weights']/alpha[hits['pixel_ids']].clamp_min(1e-8)
    matrix = torch.sparse_coo_tensor(torch.stack([hits['pixel_ids'], hits['gaussian_ids']]),
        weights, (h*w, len(model.xyz))).coalesce()
    matrices.append(matrix)
    alpha_maps.append(alpha.cpu())
    targets.append(target)
    receipts.append({'frame': frame, 'feature_sha256': rec['sha256'], 'shape': rec['shape'],
                     'intrinsic': k.tolist(), 'world_to_camera': pose.cpu().tolist()})
    if frame == training[0]:
        # Verify the sparse observation used for supervision equals the renderer.
        rows = torch.stack([torch.ones_like(model.xyz[:,0]), model.xyz[:,0]], dim=1)
        direct = renderer.render_feature_rows(model, pose, rows, alpha_normalize=True)['feature_map'].reshape(2,-1).T
        sparse = torch.sparse.mm(matrix, rows)
        transport_error = float((direct-sparse).abs().max())
        assert transport_error < 2e-4, transport_error
train = matrices[:-1]
mass = sum(torch.sparse.sum(x, dim=0).to_dense() for x in train)
seen = mass > 1e-6
hold_support = torch.sparse.mm(matrices[-1], seen.float()[:,None]).reshape(-1).cpu()
initial, final = [[] for _ in matrices], [[] for _ in matrices]
# Solve the same alpha-normalized observation model used for 2D reading.
# Channel chunks bound memory, without projection or dropping RADIO channels.
for start in range(0, d, 128):
    ys = [t[:,start:start+128].to(device) for t in targets]
    rhs = sum(torch.sparse.mm(x.T, y) for x,y in zip(train, ys))
    field = rhs/mass.clamp_min(1e-6)[:,None]
    for i,x in enumerate(matrices):
        initial[i].append(torch.sparse.mm(x,field).cpu())
    for step in range(20):
        residual = sum(torch.sparse.mm(x.T, y-torch.sparse.mm(x,field)) for x,y in zip(train,ys))
        field += .5*residual/mass.clamp_min(1e-6)[:,None]
    for i,x in enumerate(matrices):
        final[i].append(torch.sparse.mm(x,field).cpu())
    print(f'{a.scene}: channels {start}:{min(start+128,d)} complete', flush=True)
metrics = []
for i,(before,after,target) in enumerate(zip(initial,final,targets)):
    before, after = torch.cat(before,1), torch.cat(after,1)
    valid = alpha_maps[i] > .1
    supported = valid & (hold_support > .5) if i == len(training) else valid
    entry = {'frame': (training+[heldout])[i], 'split': 'heldout' if i == len(training) else 'train',
        'geometry_coverage': float(valid.float().mean()), 'evaluated_coverage': float(supported.float().mean())}
    for name,value in [('initial',before),('reconstructed',after)]:
        cosine = F.cosine_similarity(value,target,dim=1)
        entry[name+'_cosine_supported'] = float(cosine[supported].mean()) if supported.any() else None
        entry[name+'_cosine_all_pixels'] = float(cosine.mean())
        entry[name+'_mse_supported'] = float((value[supported]-target[supported]).square().mean()) if supported.any() else None
    metrics.append(entry)
model.assert_binding(binding)
output = {'role':'uncompressed_full_radio_source_reconstruction_diagnostic',
    'scene':a.scene,'geometry_report_sha256':sha256_file(report_path),'geometry_sha256':g['geometry_sha256'],
    'feature_manifest_sha256':sha256_file(manifest_path),'features':receipts,
    'training_frames':training,'heldout_frames':[heldout], 'channels':d,'iterations':20,
    'geometry_binding_unchanged':True,'renderer_sparse_max_error':transport_error,
    'observation_contract':'alpha_normalized_exact_contribution_v1',
    'feature_contract':'per_pixel_l2_normalized_raw_backbone',
    'trained_row_count':int(seen.sum()),'metrics':metrics,
    'limitations':['four temporal training views, one heldout source view',
        'uncompressed raw field; no D512 compression or capability preservation claim',
        'legacy re-sealed backbone provenance; no newly extracted native adaptor maps',
        'no benchmark labels, queries, or historical per-Gaussian features used']}
a.output.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
