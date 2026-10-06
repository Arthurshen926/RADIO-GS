"""Encode every nonempty authorized source region in one native feature space."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--crop-policy', choices=['masked', 'natural'], default='masked',
                        help='Same bbox+3px; natural retains RGB outside the region inside that box')
    args = parser.parse_args()
    torch.set_num_threads(4)
    scene = json.loads(Path(args.scene).read_text())
    authority = json.loads(Path(scene['source_rgb_authority']).read_text())
    paths = {int(r['image_id'].split('_')[-1]): r['path'] for r in authority['images']}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = NativeSiglip2Runtime.load(Path(scene['text_encoder']), device=torch.device('cuda'))
    records = []
    for frame in scene['source_frames']:
        providers = [torch.load(Path(directory) / f'frame_{frame:05d}.pt', map_location='cpu', weights_only=False)
                     for directory in scene['native_region_directories']]
        sam_path = providers[0]['sam_path']
        sam = torch.load(sam_path, map_location='cpu', weights_only=False)
        masks = unpack_masks(sam['packed_masks'], sam['mask_shape'][1])
        cached = {}
        for provider in providers:
            policy = provider.get('crop_policy', 'masked' if provider.get('crop', '').startswith('masked RGB') else None)
            if provider['sam_path'] == sam_path and provider['encoder']['path'] == scene['text_encoder'] and policy == args.crop_policy:
                cached.update({int(index): vector for index, vector in
                               zip(provider['native_proposal_indices'], provider['descriptors'])})
        ids = np.flatnonzero(masks.sum(axis=(1, 2)) > 0).tolist()
        missing = [index for index in ids if index not in cached]
        rgb = np.array(Image.open(paths[frame]).convert('RGB'))
        height, width = rgb.shape[:2]
        for begin in range(0, len(missing), 8):
            batch, indices = [], missing[begin:begin + 8]
            for index in indices:
                mask = masks[index]
                ys, xs = np.where(mask)
                x0, x1 = max(0, int(xs.min()) - 3), min(width, int(xs.max()) + 4)
                y0, y1 = max(0, int(ys.min()) - 3), min(height, int(ys.max()) + 4)
                crop = rgb[y0:y1, x0:x1].copy()
                if args.crop_policy == 'masked':
                    crop[~mask[y0:y1, x0:x1]] = 127
                image = Image.fromarray(crop).resize((384, 384), Image.Resampling.BICUBIC)
                batch.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255)
            vectors = F.normalize(runtime.encode(torch.stack(batch)).float(), dim=-1).half().cpu()
            cached.update(zip(indices, vectors))
        descriptors = torch.stack([cached[index] for index in ids]) if ids else torch.empty(0, 1536, dtype=torch.float16)
        torch.save(dict(frame=frame, descriptors=descriptors, native_proposal_indices=torch.tensor(ids),
                        sam_path=sam_path, native_mask_hw=[height, width],
                        encoder=dict(path=scene['text_encoder'], native=True), query_independent=True,
                        crop_policy=args.crop_policy,
                        crop=('masked RGB bbox+3pixels, gray127 background,384' if args.crop_policy == 'masked'
                              else 'natural RGB bbox+3pixels, unmodified background,384'), area_filter=None),
                   output / f'frame_{frame:05d}.pt')
        records.append(dict(frame=frame, regions=len(ids), new_encodings=len(missing),
                            reused_encodings=len(ids) - len(missing)))
        print(scene['scene'], frame, records[-1], flush=True)
    (output / 'encoding.json').write_text(json.dumps(dict(scene=args.scene, frames=scene['source_frames'],
        source_frame_cap=None, region_area_filter=None, empty_masks_excluded=True, records=records,
        crop_policy=args.crop_policy, query_or_target_labels_used=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
