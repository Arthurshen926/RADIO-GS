"""Source-only real-asset check of MoGe-conditioned Gaussian writes, no queries."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import marginal_responsibility_statistics
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--fusion-root', type=Path, required=True)
parser.add_argument('--primitive-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
results = []
for scene in ('figurines', 'ramen', 'teatime', 'waldo_kitchen'):
    path = args.fusion_root / (scene + '.json')
    prior = MogeGeometryPrior(str(path), sha256_file(path))
    source = next(r for r in prior.report['geometry_receipt']['inputs'] if r['role'] == 'source_authority')
    authority_path = Path(source['path'])
    authority = json.loads(authority_path.read_text())
    field_path = args.primitive_root / scene / 'primitive_query_method_v1.pth'
    field = torch.load(field_path, map_location='cpu', weights_only=False)
    xyz = field['xyz'].float().cpu()
    digest = hashlib.sha256(xyz.contiguous().numpy().astype('<f4').tobytes()).hexdigest()
    assert digest == authority['metadata']['xyz_sha256']
    row = next(r for r in authority['views'] if prior.calibration[r['frame_index']]['accepted'])
    view_path = authority_path.parent / row['relative_path']
    assert sha256_file(view_path) == row['sha256']
    view = torch.load(view_path, map_location='cpu', weights_only=False)
    camera = prior.cameras[row['frame_index']]
    h, w = authority['metadata']['feature_height'], authority['metadata']['feature_width']
    config = load_config(authority['metadata']['config'])
    intrinsic = torch.tensor([[config.fx*w/config.image_width, 0, config.cx*w/config.image_width],
                              [0, config.fy*h/config.image_height, config.cy*h/config.image_height],
                              [0, 0, 1]], dtype=torch.float32)
    intrinsic_digest = hashlib.sha256(intrinsic.numpy().astype('<f4').tobytes()).hexdigest()
    assert intrinsic_digest == authority['metadata']['intrinsics_sha256']
    weights = marginal_responsibility_statistics(view['pixel_ids'].long(), view['base_weights'], num_pixels=h*w).target_weight
    original = weights.clone()
    kwargs = dict(frame_index=row['frame_index'], xyz=xyz, world_to_camera=camera.world_to_camera,
                  intrinsic=intrinsic, gaussian_ids=view['gaussian_ids'], pixel_ids=view['pixel_ids'],
                  weights=weights, height=h, width=w)
    revised = prior.apply(**kwargs)
    assert torch.equal(weights, original)
    assert bool((revised >= .5 * weights).all()) and bool((revised <= weights).all())
    prior.strength = 0
    assert torch.equal(prior.apply(**kwargs), original)
    results.append({'scene':scene, 'role':'source_write_diagnostic_not_end_to_end_metric',
                    'field_sha256':sha256_file(field_path), 'source_authority_sha256':source['sha256'],
                    'xyz_matches_exact_gaussian_authority':True, 'disabled_bit_exact':True,
                    'original_authority_unchanged':True, 'write_statistics':prior.statistics[0]})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
