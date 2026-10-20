"""SAM3 worker: stream original full-menu detections through a bounded pipe.

Run in the SAM3 environment. Stdout is a length-prefixed torch payload stream;
diagnostics go to stderr. No RGB, masks, features or predictions are saved.
"""
import argparse
import contextlib
import io
import struct
import sys
from pathlib import Path
import json

import torch
from torch.nn import functional as F
from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model, sam3_autocast_context
from radio_gs.scannet_constants import NYU40_ID_TO_NAME


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--scenes', nargs='+', required=True)
    args = p.parse_args(); config = json.loads(Path(args.config).read_text()); stream = sys.stdout.buffer
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(sys.stderr):
        processor = _load_sam3_model(checkpoint_path='/root/RADIO-GS/checkpoints/sam3_modelscope/sam3.pt', device='cuda',
            confidence_threshold=.5, dtype='float32', resolution=1008, build_on_cpu=True)
        with torch.inference_mode(), sam3_autocast_context('cuda', torch.bfloat16):
            for scene in args.scenes:
                spec = json.loads((Path(config['source'])/'maps'/scene/'map.json').read_text())
                sensor = SensorIndex(Path(config['raw_source'])/'raw'/scene/(scene+'.sens'))
                for frame in spec['source_frames']:
                    _, _, rgb = sensor.read(frame); state = processor.set_image(rgb)
                    rows = []
                    for c in config['class_ids']:
                        processor.reset_all_prompts(state)
                        prediction = processor.set_text_prompt(NYU40_ID_TO_NAME[c], state)
                        scores = prediction['scores'].float().cpu()
                        masks = (F.interpolate(prediction['masks_logits'].float().reshape(len(scores), 1,
                            *prediction['masks_logits'].shape[-2:]), (96, 128), mode='bilinear', align_corners=False)[:, 0].cpu().half()
                            if len(scores) else torch.empty(0, 96, 128, dtype=torch.float16))
                        rows.append(dict(scores=scores, masks=masks))
                    payload = io.BytesIO(); torch.save(dict(scene=scene, frame=frame, rows=rows), payload)
                    data = payload.getvalue()
                    if len(data) > 32*2**20:
                        raise ValueError('one source view exceeds bounded pipe payload')
                    stream.write(struct.pack('<Q', len(data))); stream.write(data); stream.flush()


if __name__ == '__main__':
    main()
