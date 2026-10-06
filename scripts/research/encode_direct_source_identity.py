"""Measure native identity on the detector's own mask, not a nearby SAM region."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from transformers import SiglipVisionModel
from radio_gs.training.small_region_inventory import original_native_gray_crop


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--context-preserving', action='store_true', help='Unmasked 1.5x context for identity only; detector scope is unchanged')
    a = p.parse_args()
    torch.set_num_threads(4)
    inputs = json.loads(Path(a.inputs).read_text())
    scene = json.loads(Path(inputs['scene']).read_text())
    response = torch.load(inputs['response'], map_location='cpu', weights_only=False)
    queries = F.normalize(response['encodings'].float(), dim=-1).cuda()
    paths = {int(r['image_id'].split('_')[-1]): r['path']
             for r in json.loads(Path(scene['source_rgb_authority']).read_text())['images']}
    model = SiglipVisionModel.from_pretrained(scene['text_encoder'], local_files_only=True,
                                              torch_dtype=torch.float16).cuda().eval()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    matches = {}
    instance_count, empty_count = 0, 0
    start = time.monotonic()
    for frame in scene['source_frames']:
        data = torch.load(Path(inputs['detections']) / f'identity_{frame}.pt', map_location='cpu', weights_only=False)
        rgb = np.asarray(Image.open(paths[frame]).convert('RGB'))
        crops, indices = [], []
        rows = []
        for qi, record in enumerate(data['instances']):
            row = []
            for mi, mask in enumerate(record['masks']):
                binary = np.array(Image.fromarray((mask.float().numpy() > .5)).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST), copy=True)
                valid = bool(binary.any())
                row.append(dict(identity_score_valid=valid, native_score=None,
                                descriptor_source='detector_context_rgb_crop' if a.context_preserving else 'same_detector_mask_native_rgb_crop'))
                instance_count += 1
                if not valid:
                    empty_count += 1
                    continue
                if a.context_preserving:
                    ys, xs = np.where(binary)
                    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
                    w, h = 1.5 * (xs.max() - xs.min() + 1), 1.5 * (ys.max() - ys.min() + 1)
                    box = (max(0, math.floor(cx - w / 2)), max(0, math.floor(cy - h / 2)),
                           min(rgb.shape[1], math.ceil(cx + w / 2)), min(rgb.shape[0], math.ceil(cy + h / 2)))
                    crop = Image.fromarray(rgb).crop(box).resize((384, 384), Image.Resampling.BICUBIC)
                else:
                    crop = original_native_gray_crop(rgb, binary)
                crops.append(pil_to_tensor(crop).float() / 255)
                indices.append((qi, mi))
            rows.append(row)
        descriptors = []
        for offset in range(0, len(crops), 8):
            values = ((torch.stack(crops[offset:offset + 8]).cuda() - .5) / .5).half()
            features = F.normalize(model(pixel_values=values).pooler_output.float(), dim=-1)
            ids = torch.tensor([q for q, _ in indices[offset:offset + 8]], device='cuda')
            similarity = (features * queries[ids]).sum(-1).cpu().tolist()
            for (qi, mi), score in zip(indices[offset:offset + 8], similarity):
                rows[qi][mi]['native_score'] = score
            descriptors.append(features.cpu().half())
        torch.save(dict(frame=frame, instance_indices=indices,
                        descriptors=torch.cat(descriptors) if descriptors else torch.empty(0, queries.shape[1])),
                   out / f'features_{frame}.pt')
        matches[frame] = rows
        print(scene['scene'], frame, len(indices), round(time.monotonic() - start, 1), flush=True)
    result = dict(queries=[dict(query=q) for q in response['texts']], matches=matches,
                  heldout_stride=0, identity_measurement='direct_detector_mask_native_siglip2',
                  source_frames=scene['source_frames'], original_inputs=a.inputs,
                  instances=instance_count, encoded_instances=instance_count - empty_count,
                  empty_masks_unknown=empty_count, native_encoder=scene['text_encoder'],
                  crop='unmasked detector bbox1.5x,384bicubic; identity only, never added scope' if a.context_preserving else 'same existing masked RGB bbox±3,gray127,384bicubic recipe',
                  target_rgb_used=False, seconds=time.monotonic() - start)
    (out / 'identity.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
