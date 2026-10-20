"""Frozen descriptor/query transport and SAM response worker, without GT paths."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.querying.image_conditioned_response import response_prompt
from radio_gs.querying.signed_reference_affinity import patch_authority,signed_reference_probability
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model


def binary(path):
    value=np.asarray(Image.open(path));return torch.from_numpy(np.any(value>127,axis=2) if value.ndim==3 else value>127)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);args=p.parse_args();root=Path(args.request).resolve()
    if root.parent!=Path('/dev/shm') or root.stat().st_mode&0o077:raise ValueError('private tmpfs required')
    r=json.loads((root/'request.json').read_text());torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(r['gpu_memory_fraction'])
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in r['source_paths']}
    prior=np.load(root/'prior.npy',allow_pickle=False);h,w=prior.shape;assert (h,w)==(756,1008)
    pos,neg,seed=[binary(r[k]) for k in ['positive_scribble','negative_scribble','reference_mask']]
    assert pos.shape==neg.shape==seed.shape==prior.shape and pos.any() and neg.any()
    sys.path.insert(0,r['dino_repository']);from dinov2.hub.backbones import dinov2_vitb14
    dino=dinov2_vitb14(pretrained=False);dino.load_state_dict(torch.load(r['dino_checkpoint'],map_location='cpu',weights_only=True),strict=True);dino.eval().cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
    features=[]
    for path in [r['reference_rgb'],root/'rgb.png']:
        image=Image.open(path).convert('RGB').resize((w,h),Image.Resampling.BILINEAR)
        x=torch.from_numpy(np.array(image)).permute(2,0,1).cuda().float()/255
        with torch.autocast('cuda',dtype=torch.bfloat16):out=dino.forward_features(((x-mean)/std)[None])['x_norm_patchtokens'][0]
        features.append(out.float());assert out.shape==(h//14*(w//14),768)
    probs={};records={}
    for arm,use_seed in [('signed',False),('pseudo',True)]:
        labels=patch_authority(pos,neg,(h//14,w//14),seed if use_seed else None).cuda()
        prob,record=signed_reference_probability(*features,labels)
        probs[arm]=F.interpolate(prob.reshape(1,1,h//14,w//14),size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        records[arm]=record;np.save(root/(arm+'_affinity.npy'),probs[arm],allow_pickle=False)
    del dino,features,x,out,prob,labels;gc.collect();torch.cuda.empty_cache()
    sam=_load_sam3_model(checkpoint_path=r['checkpoint'],device='cuda',confidence_threshold=0,dtype='bfloat16',resolution=1008,point_only=True,build_on_cpu=True)
    size=tuple(sam.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size)
    decoded={}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        state=sam.set_image(Image.open(root/'rgb.png').convert('RGB'))
        for arm,prob in [('static',prior),('signed',probs['signed']),('pseudo',probs['pseudo'])]:
            prompt=response_prompt(prob,size)
            fallback=arm!='static' and (records[arm]['fallback'] is not None or prompt is None)
            if fallback:
                output=decoded['static'];quality=None
            elif prompt is None:
                output=prior;quality=None
            else:
                logits,quality,_=sam.model.predict_inst(state,**prompt,multimask_output=False,return_logits=True)
                assert logits.shape==(1,h,w)
                output=torch.from_numpy(logits[0].astype(np.float32)).sigmoid().numpy();quality=float(np.asarray(quality).reshape(-1)[0])
            assert output.shape==prior.shape and np.isfinite(output).all()
            decoded[arm]=output;np.save(root/(arm+'_image.npy'),output,allow_pickle=False)
            records.setdefault(arm,{}).update(predicted_quality=quality,static_fallback=fallback,prompt_box=None if prompt is None else prompt['box'].tolist())
    (root/'result.json').write_text(json.dumps(dict(decoder=records,source_hashes=hashes,target_gt_available=False,
        dino_eval=True,sam_eval=not sam.model.training,negative_scribble_used=True,feature_grid=[h//14,w//14],
        torch_version=torch.__version__,peak_gpu_bytes=torch.cuda.max_memory_allocated()),indent=2)+'\n')


if __name__=='__main__':main()
