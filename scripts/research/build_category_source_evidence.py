"""Cold-scene category evidence from explicit source RGB and frozen geometry.

Run detection with detect_consolidated_sources.py at 96x128 first. This stage
adds native foreground/context identity evidence and matching alpha operators.
It reads no historical scene head, final prediction or semantic label.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.native_query_encoders import load_siglip_text
from radio_gs.querying.query_region_evidence import same_window
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--identity-from', help='Reuse paired low-resolution proposal identities; only rebuild geometry projection')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic()
    scene = json.loads(Path(args.scene).read_text())
    texts = [q['text'] for q in json.loads(Path(args.queries).read_text())['queries']]
    cameras = {r['frame_index']: r for r in json.loads(Path(scene['source_cameras']).read_text())['records']}
    if not args.identity_from:
        encoder, tokenizer = load_siglip_text(scene['text_encoder'], 'cuda')
        tokens = tokenizer(texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
        query = F.normalize(F.normalize(encoder(**tokens).pooler_output.float(), dim=-1).half().float(), dim=-1)
        del encoder
        torch.cuda.empty_cache()
        runtime = NativeSiglip2Runtime.load(Path(scene['text_encoder']), device=torch.device('cuda'))
    geometry = FrozenGaussianGeometry.load(scene['geometry']['path'], expected_sha256=scene['geometry']['sha256'], device='cuda')
    out = Path(args.output)
    records = []
    for frame in scene['source_frames'][args.shard::args.shards]:
        camera = cameras[frame]
        data = torch.load(out/'detections'/f'identity_{frame}.pt', map_location='cpu', weights_only=False)
        h, w = data['raster_hw']
        if args.identity_from:
            paired = torch.load(Path(args.identity_from)/f'identity_{frame}.pt', map_location='cpu', weights_only=False)
            for row, old in zip(data['instances'], paired['instances']):
                row['foreground_margin'] = old['foreground_margin']
        else:
            rgb = torch.from_numpy(np.array(Image.open(camera['source_image']).convert('RGB'), copy=True)).permute(2, 0, 1).float()/255
            # Encode one proposal at a time: fixed teacher computation, no scene-size
            # memory growth from retaining all full/foreground/background crops.
            for qi, row in enumerate(data['instances']):
                row['foreground_margin'] = torch.full((len(row['scores']),), -1.)
                for j, mask in enumerate(row['masks']):
                    if not (mask > .5).any():
                        continue
                    full_mask = F.interpolate((mask > .5)[None, None].float(), size=rgb.shape[1:], mode='nearest-exact')[0, 0].bool()
                    triple, _ = same_window(rgb, full_mask)
                    feature = F.normalize(runtime.encode(triple).float(), dim=-1)
                    row['foreground_margin'][j] = (feature[1] @ query[qi] - feature[2] @ query[qi]).cpu()
        if args.identity_from:
            data['boundary_radius'] = round(w / paired['raster_hw'][1]) * paired.get('boundary_radius', 1)
        torch.save(data, out/f'identity_{frame}.pt')
        k = torch.as_tensor(camera['intrinsic'], device='cuda', dtype=torch.float32).clone()
        k[0] *= w / camera['width']
        k[1] *= h / camera['height']
        renderer = SimpleNamespace(scaled_intrinsics=lambda width, height: k, near_plane=.001, far_plane=100.)
        hits = rasterize_single_view_contributions(geometry, renderer, torch.as_tensor(camera['world_to_camera'], device='cuda', dtype=torch.float32), height=h, width=w, include_depths=False)
        torch.save(dict(pixel_ids=hits['pixel_ids'].cpu(), gaussian_ids=hits['gaussian_ids'].cpu(), base_weights=hits['weights'].float().cpu()), out/f'contributions_{frame}.pt')
        records.append(dict(frame=frame, identity=str(out/f'identity_{frame}.pt'), contributions=str(out/f'contributions_{frame}.pt'), source_rgb=camera['source_image']))
        print(scene['scene'], frame, len(hits['weights']), round(time.monotonic()-start, 1), flush=True)
    (out/f'views_{args.shard}.json').write_text(json.dumps(dict(views=records, seconds=time.monotonic()-start, historical_scene_parameters=False, target_rgb_used=False, semantic_labels_used=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
