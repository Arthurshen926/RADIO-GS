"""Exact-query source grounding at a fixed second image scale.

Four overlapping 60%-width/height viewports visit every declared source image.
Stored masks are projected to the existing source raster. Viewport bounds remain
attached to each instance: pixels outside a crop were never observed by it.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model, sam3_autocast_context


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--queries', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--checkpoint', default='checkpoints/sam3_modelscope/sam3.pt')
    a = p.parse_args()
    torch.set_num_threads(4)
    scene = json.loads(Path(a.scene).read_text())
    texts = [q['text'] for q in json.loads(Path(a.queries).read_text())['queries']]
    images = {int(r['image_id'].split('_')[-1]): r['path']
              for r in json.loads(Path(scene['source_rgb_authority']).read_text())['images']}
    frames = scene['source_frames'][a.shard::a.shards]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    processor = _load_sam3_model(checkpoint_path=a.checkpoint, device='cuda',
                                confidence_threshold=.5, dtype='float32', resolution=1008)
    height, width = 192, 256
    yy, xx = torch.meshgrid((torch.arange(height, device='cuda') + .5) / height,
                            (torch.arange(width, device='cuda') + .5) / width, indexing='ij')
    counts = torch.zeros(len(texts), dtype=torch.long)
    start = time.monotonic()
    with sam3_autocast_context('cuda', torch.bfloat16):
        for frame in frames:
            rgb = Image.open(images[frame]).convert('RGB')
            masks = [[] for _ in texts]
            scores = [[] for _ in texts]
            viewports = [[] for _ in texts]
            for top in [0., .4]:
                for left in [0., .4]:
                    box = (round(left * rgb.width), round(top * rgb.height),
                           round((left + .6) * rgb.width), round((top + .6) * rgb.height))
                    x0, y0, x1, y1 = box[0] / rgb.width, box[1] / rgb.height, box[2] / rgb.width, box[3] / rgb.height
                    grid = torch.stack([2 * (xx - x0) / (x1 - x0) - 1,
                                        2 * (yy - y0) / (y1 - y0) - 1], -1)[None]
                    visible = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
                    state = processor.set_image(rgb.crop(box))
                    for qi, text in enumerate(texts):
                        processor.reset_all_prompts(state)
                        result = processor.set_text_prompt(text, state)
                        n = len(result['scores'])
                        if not n:
                            continue
                        # SAM3's masks_logits field contains sigmoid probabilities.
                        probability = result['masks_logits'].float().reshape(n, 1, *result['masks_logits'].shape[-2:])
                        projected = F.grid_sample(probability, grid.expand(n, -1, -1, -1),
                                                  mode='bilinear', padding_mode='zeros', align_corners=False)[:, 0]
                        projected *= visible
                        masks[qi].append(projected.cpu().half())
                        scores[qi].append(result['scores'].float().cpu())
                        viewports[qi].append(torch.tensor([[x0, y0, x1, y1]]).repeat(n, 1))
            instances = []
            for qi in range(len(texts)):
                present = bool(scores[qi])
                counts[qi] += present
                instances.append(dict(scores=torch.cat(scores[qi]) if present else torch.empty(0),
                    masks=torch.cat(masks[qi]) if present else torch.empty(0, height, width, dtype=torch.float16),
                    viewports=torch.cat(viewports[qi]) if present else torch.empty(0, 4)))
            torch.save(dict(frame=frame, queries=texts, instances=instances,
                            provider='four_fixed_source_viewports_exact_text'), out / f'identity_{frame}.pt')
            print(scene['scene'], frame, round(time.monotonic() - start, 1), flush=True)
    (out / f'provider_{a.shard}.json').write_text(json.dumps(dict(scene=a.scene, queries=a.queries,
        source_frames=frames, shard=a.shard, shards=a.shards, crop_scale=.6, crop_origins=[0., .4],
        detected_views=dict(zip(texts, counts.tolist())), checkpoint=a.checkpoint, teacher_resolution=1008,
        image_encoder_calls=4 * len(frames), text_grounding_calls=4 * len(frames) * len(texts),
        target_rgb_used=False, query_rewrites=False, original_source_pixels=True,
        seconds=time.monotonic() - start), indent=2) + '\n')


if __name__ == '__main__':
    main()
