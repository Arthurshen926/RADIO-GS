"""SAM3-only worker: no dataset, target GT, geometry or task-name arguments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from radio_gs.querying.image_conditioned_response import response_prompt
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    args = p.parse_args()
    root = Path(args.request).resolve()
    if root.parent != Path('/dev/shm') or root.stat().st_mode & 0o077:
        raise ValueError('worker accepts a private tmpfs request only')
    request = json.loads((root / 'request.json').read_text())
    source_hashes = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in request['source_paths']}
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(request['gpu_memory_fraction'])
    model = _load_sam3_model(checkpoint_path=request['checkpoint'], device='cuda',
        confidence_threshold=0, dtype='bfloat16', resolution=1008, point_only=True, build_on_cpu=True)
    size = tuple(model.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size)
    prior = np.load(root / 'prior.npy', allow_pickle=False)
    prompt = response_prompt(prior, size)
    rows = {}
    for mode in ('render', 'rgb'):
        image = Image.open(root / (mode + '.png')).convert('RGB')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = model.set_image(image)
            for kind in ('box', 'mask'):
                arm = mode + '_' + kind
                if prompt is None:
                    output = torch.logit(torch.from_numpy(prior).clamp(1e-4, 1-1e-4)).numpy()
                    quality = None
                else:
                    kwargs = {'box': prompt['box']}
                    if kind == 'mask':
                        kwargs['mask_input'] = prompt['mask_input']
                    output, quality, _ = model.model.predict_inst(state, **kwargs, multimask_output=False, return_logits=True)
                    assert output.shape == (1, *prior.shape)
                    output = output[0].astype(np.float32)
                    quality = float(np.asarray(quality).reshape(-1)[0])
                assert np.isfinite(output).all()
                np.save(root / (arm + '.npy'), output, allow_pickle=False)
                rows[arm] = dict(predicted_quality=quality, fallback_empty_prompt=prompt is None)
        del state
    result = dict(decoder=rows, mask_input_hw=list(size),
                  prompt_box=None if prompt is None else prompt['box'].tolist(),
                  torch_version=torch.__version__, target_gt_available=False, source_hashes=source_hashes)
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
