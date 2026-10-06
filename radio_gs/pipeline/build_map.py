"""Reconstruct a native map from explicit geometry, cameras and teacher regions."""
import argparse
import json
from pathlib import Path
import time

import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from .native_reconstruction import NativeFieldAccumulator


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='Explicit geometry, cameras, region directories and calibration parameters')
    parser.add_argument('--regions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--region-policy', choices=['mixture', 'finest'], default='mixture')
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    scene = json.loads(Path(args.scene).read_text())
    binding = scene['geometry']
    geometry = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    prior = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
    accumulator = NativeFieldAccumulator(geometry, scene['native_dimension'], region_policy=args.region_policy)
    height, width = scene.get('source_fit_hw', [192, 256])
    records = []
    for frame in scene['source_frames']:
        row = torch.load(Path(args.regions) / f'frame_{frame:05d}.pt', map_location='cpu', weights_only=False)
        sam = torch.load(row['sam_path'], map_location='cpu', weights_only=False)
        masks = unpack_masks(sam['packed_masks'], sam['mask_shape'][1])[row['native_proposal_indices'].numpy()]
        camera = prior.cameras[frame]
        record = accumulator.add_view(masks.copy(), row['descriptors'], camera.intrinsic, camera.world_to_camera,
                                      height=height, width=width, image_height=camera.height, image_width=camera.width,
                                      near_plane=.001 * prior.scale, far_plane=100 * prior.scale)
        records.append(dict(frame=frame, **record))
        print(scene['scene'], frame, record, flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    field = accumulator.field(binding, scene['native_space'], scene['response_calibration'])
    torch.save(field, output / 'native_field.pt')
    deployed = dict(scene, region_policy=args.region_policy, native_field=str(output / 'native_field.pt'), native_region_directories=[args.regions])
    (output / 'scene.json').write_text(json.dumps(deployed, indent=2) + '\n')
    (output / 'build.json').write_text(json.dumps(dict(scene=args.scene, records=records,
        source_frames=scene['source_frames'], region_policy=args.region_policy, seconds=time.monotonic() - started,
        geometry_trainable=False, query_independent_feature_write=True,
        source_calibration_frozen=scene['response_calibration'], old_field_features_read=False,
        query_predictions_or_evaluation_labels_read=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
