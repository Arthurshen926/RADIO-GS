"""Frozen SAM3 worker for matched partial-response prompts; no target GT paths."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from radio_gs.querying.partial_response_prompt import partial_response_prompts
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
    root=Path(a.request).resolve()
    if root.parent!=Path('/dev/shm') or root.stat().st_mode&0o077:raise ValueError('private tmpfs required')
    request=json.loads((root/'request.json').read_text())
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in request['source_paths']}
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(request['gpu_memory_fraction'])
    model=_load_sam3_model(checkpoint_path=request['checkpoint'],device='cuda',confidence_threshold=0,
        dtype='bfloat16',resolution=1008,point_only=True,build_on_cpu=True)
    hw=tuple(model.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size)
    prior=np.load(root/'prior.npy',allow_pickle=False)
    variants=partial_response_prompts(prior,hw,request['point_budget'])
    rows={};image=Image.open(root/'rgb.png').convert('RGB')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        state=model.set_image(image)
        for arm in ('signed_core','positive_core','sparse_core'):
            if variants is None:
                output=torch.logit(torch.from_numpy(prior).clamp(1e-4,1-1e-4)).numpy();quality=None
            else:
                output,quality,_=model.model.predict_inst(state,**variants[arm],multimask_output=False,return_logits=True)
                assert output.shape==(1,*prior.shape)
                output=output[0].astype(np.float32);quality=float(np.asarray(quality).reshape(-1)[0])
            assert np.isfinite(output).all()
            np.save(root/(arm+'.npy'),output,allow_pickle=False)
            rows[arm]=dict(predicted_quality=quality,fallback_empty_prompt=variants is None)
    common=None if variants is None else {k:variants['sparse_core'][k].tolist() for k in ('box','point_coords','point_labels')}
    (root/'result.json').write_text(json.dumps(dict(decoder=rows,mask_input_hw=list(hw),common_prompt=common,
        source_hashes=hashes,torch_version=torch.__version__,target_gt_available=False),indent=2)+'\n')


if __name__=='__main__':main()
