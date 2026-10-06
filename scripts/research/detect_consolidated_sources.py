"""Exact runtime text on every declared source image; no legacy query artifacts."""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model, sam3_autocast_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--paired-low-output', help='Save the same detector forward at 96x128 for a matched control')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    scene = json.loads(Path(args.scene).read_text())
    queries = json.loads(Path(args.queries).read_text())['queries']
    if any(q['modality'] != 'text' or q['intent'] != 'category' for q in queries):
        raise ValueError('This evidence provider implements text category sets only')
    texts = [q['text'] for q in queries]
    authority = json.loads(Path(scene['source_rgb_authority']).read_text())
    images = {int(row['image_id'].split('_')[-1]): row['path'] for row in authority['images']}
    processor = _load_sam3_model(checkpoint_path=args.checkpoint, device='cuda',
                                confidence_threshold=.5, dtype='float32', resolution=1008)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    low_out = Path(args.paired_low_output) if args.paired_low_output else None
    if low_out: low_out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    counts = [0] * len(texts)
    with torch.inference_mode(), sam3_autocast_context('cuda', torch.bfloat16):
        for frame in scene['source_frames'][args.shard::args.shards]:
            state = processor.set_image(Image.open(images[frame]).convert('RGB'))
            instances = []
            low_instances = []
            for j, text in enumerate(texts):
                processor.reset_all_prompts(state)
                detection = processor.set_text_prompt(text, state)
                scores = detection['scores'].float().cpu()
                if len(scores):
                    masks = F.interpolate(detection['masks_logits'].float().reshape(
                        len(scores), 1, *detection['masks_logits'].shape[-2:]),
                        size=(args.height, args.width), mode='bilinear', align_corners=False)[:, 0].cpu().half()
                    counts[j] += 1
                else:
                    masks = torch.empty(0, args.height, args.width, dtype=torch.float16)
                instances.append(dict(scores=scores, masks=masks))
                if low_out:
                    low_masks = F.interpolate(detection['masks_logits'].float().reshape(len(scores), 1, *detection['masks_logits'].shape[-2:]), size=(96, 128), mode='bilinear', align_corners=False)[:, 0].cpu().half() if len(scores) else torch.empty(0, 96, 128, dtype=torch.float16)
                    low_instances.append(dict(scores=scores, masks=low_masks))
            torch.save(dict(queries=texts, instances=instances, frame=frame,
                            mask_tensor_semantics='probability',
                            raster_hw=[args.height, args.width],
                            provider='exact_text_sam3_no_historical_anchor'), out/f'identity_{frame}.pt')
            if low_out:
                torch.save(dict(queries=texts, instances=low_instances, frame=frame, mask_tensor_semantics='probability', raster_hw=[96, 128], provider='exact_text_sam3_no_historical_anchor'), low_out/f'identity_{frame}.pt')
            print(scene['scene'], frame, round(time.monotonic()-start, 1), flush=True)
    report = 'provider.json' if args.shards == 1 else f'provider_{args.shard}.json'
    (out/report).write_text(json.dumps(dict(
        scene=args.scene, queries=args.queries, checkpoint=args.checkpoint,
        source_frames=scene['source_frames'][args.shard::args.shards], detected_views=dict(zip(texts, counts)),
        raster_hw=[args.height, args.width], mask_tensor_semantics='probability',
        historical_predictions_read=False, target_rgb_used=False,
        identity_confidence='raw SAM3 detection score; not correctness probability',
        scope='within-view category union; no cross-view instance identity assertion',
        paired_low_output=str(low_out) if low_out else None, image_forwards_shared_between_rasters=bool(low_out),
        seconds=time.monotonic()-start), indent=2)+'\n')


if __name__ == '__main__':
    main()
