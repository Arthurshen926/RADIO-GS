"""Reference-initialized SAM3 tracking over explicitly authorized source frames."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from sam3.model_builder import build_sam3_video_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    scene = json.loads(Path(args.scene).read_text())
    query = json.loads(Path(args.query).read_text())
    frames = {frame: scene['source_rgb'][str(frame)] for frame in scene['tracking_frames']}
    order = sorted(frames)
    reference = query['reference_frame']
    seed_index = order.index(reference)
    points = query.get('points_normalized')
    if points is None:
        seed = np.asarray(Image.open(query['reference_mask']))
        seed = np.any(seed > 127, axis=2) if seed.ndim == 3 else seed > 127
    model = build_sam3_video_model(
        checkpoint_path=scene['checkpoint'], load_from_HF=False,
        strict_state_dict_loading=True, apply_temporal_disambiguation=True,
        device='cuda', compile=False,
    )
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    predictions = {}
    with tempfile.TemporaryDirectory(prefix='radio_source_video_') as temp, torch.autocast('cuda', dtype=torch.bfloat16):
        size = Image.open(query['reference_rgb']).size
        width = 1008
        height = round(size[1] * width / size[0])
        for index, frame in enumerate(order):
            Image.open(frames[frame]).convert('RGB').resize(
                (width, height), Image.Resampling.BILINEAR
            ).save(Path(temp) / f'{index:05d}.jpg')
        state = tracker.init_state(video_path=temp, offload_video_to_cpu=True)
        if points is None:
            tracker.add_new_mask(inference_state=state, frame_idx=seed_index, obj_id=1, mask=torch.from_numpy(seed))
        else:
            tracker.add_new_points_or_box(
                inference_state=state, frame_idx=seed_index, obj_id=1,
                points=torch.tensor(points, dtype=torch.float32),
                labels=torch.tensor(query['point_labels'], dtype=torch.int32), rel_coordinates=True,
            )
        for reverse in (False, True):
            for index, ids, _low, masks, scores in tracker.propagate_in_video(
                state, start_frame_idx=seed_index, max_frame_num_to_track=len(order),
                reverse=reverse, propagate_preflight=True,
            ):
                object_index = list(ids).index(1)
                predictions[order[int(index)]] = dict(
                    logits=masks[object_index].float().cpu().squeeze().half(),
                    object_logit=float(torch.as_tensor(scores[object_index]).cpu()),
                )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(frames=predictions, reference_frame=reference), output / 'tracks.pt')
    (output / 'tracking.json').write_text(json.dumps(dict(
        scene=args.scene, query=args.query, source_frames=order, source_rgb=frames,
        reference_frame=reference, target_rgb_used=False,
        negative_prompt_used=any(label == 0 for label in query.get('point_labels', [])),
        teacher='SAM3 temporal tracker', seed='reference mask' if points is None else 'signed normalized point prompts',
        missing_object_is_unknown=True,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
