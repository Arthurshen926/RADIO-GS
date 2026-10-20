"""Matched static image and reference-memory SAM3 inference; no GT paths."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.v2 import functional as TF
from radio_gs.querying.image_conditioned_response import response_prompt
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model


class LazyImages:
    def __init__(self,paths,wh,transform):
        self.paths,self.wh,self.transform=paths,wh,transform
    def __getitem__(self,index):
        image=Image.open(self.paths[index]).convert('RGB').resize(self.wh,Image.Resampling.BILINEAR)
        return self.transform(TF.to_image(image))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
    root=Path(a.request).resolve()
    if root.parent!=Path('/dev/shm') or root.stat().st_mode&0o077:raise ValueError('private tmpfs required')
    r=json.loads((root/'request.json').read_text());torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(r['gpu_memory_fraction'])
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in r['source_paths']}
    model=_load_sam3_model(checkpoint_path=r['checkpoint'],device='cuda',confidence_threshold=0,
        dtype='bfloat16',resolution=1008,point_only=True,build_on_cpu=True)
    tracker=model.model.inst_interactive_predictor.model
    assert not tracker.training and not model.model.training
    prior=np.load(root/'prior.npy',allow_pickle=False);h,w=prior.shape
    seed=np.asarray(Image.open(r['reference_mask']))
    assert seed.ndim==2 and seed.shape==prior.shape and (seed>127).any()
    prompt=response_prompt(prior,tuple(tracker.sam_prompt_encoder.mask_input_size))
    records={}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        state=model.set_image(Image.open(root/'rgb.png').convert('RGB'))
        if prompt is None:
            logits=torch.logit(torch.from_numpy(prior).clamp(1e-4,1-1e-4)).numpy();quality=None
        else:
            logits,quality,_=model.model.predict_inst(state,**prompt,multimask_output=False,return_logits=True)
            assert logits.shape==(1,h,w);logits=logits[0].astype(np.float32)
            quality=float(np.asarray(quality).reshape(-1)[0])
        np.save(root/'static_image.npy',logits,allow_pickle=False)
        records['static_image']=dict(predicted_quality=quality,fallback_empty_prompt=prompt is None)
        del state,logits;gc.collect();torch.cuda.empty_cache()
        tracker.backbone=model.model.backbone
        tracker.use_memory_selection=True
        for arm,order in r['paths'].items():
            assert order[0]==r['reference_frame'] and order[-1]==r['target_frame']
            state=tracker.init_state(video_height=h,video_width=w,num_frames=len(order),
                offload_video_to_cpu=True,offload_state_to_cpu=True)
            state['images']=LazyImages([r['frame_paths'][str(f)] for f in order],(w,h),model.transform)
            tracker.add_new_mask(inference_state=state,frame_idx=0,obj_id=1,mask=torch.from_numpy(seed>127))
            scores=[];last=None
            for index,ids,low,masks,presence in tracker.propagate_in_video(state,start_frame_idx=0,
                    max_frame_num_to_track=len(order),reverse=False,propagate_preflight=True):
                slot=list(ids).index(1);last=int(index)
                scores.append(float(torch.as_tensor(presence[slot]).cpu()))
                if last==len(order)-1:
                    output=masks[slot].float().cpu().squeeze().numpy()
                    assert output.shape==(h,w) and np.isfinite(output).all()
                    np.save(root/(arm+'.npy'),output,allow_pickle=False)
            assert last==len(order)-1 and len(scores)==len(order)
            records[arm]=dict(order=order,object_logits=scores,final_object_logit=scores[-1])
            del state,low,masks;gc.collect();torch.cuda.empty_cache()
    (root/'result.json').write_text(json.dumps(dict(decoder=records,source_hashes=hashes,
        torch_version=torch.__version__,target_gt_available=False,model_eval=True,
        memory_selection=True,matched_rgb_preprocessing=True,peak_gpu_bytes=torch.cuda.max_memory_allocated()),indent=2)+'\n')


if __name__=='__main__':main()
