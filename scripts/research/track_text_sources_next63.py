"""Automatic text-detection-seeded tracking on authorized source frames only."""
import argparse,json,tempfile,time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from sam3.model_builder import build_sam3_video_model

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--key',required=True);args=parser.parse_args();torch.set_num_threads(4)
    data=load(Path('output/native_rebuild_next54/mixture_masscal')/args.key/'field.pt');root=Path(data['root'])
    info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    order=sorted(info['train_frames']);paths={r['frame_idx']:Path(src['image_dir'])/r['source_file'] for r in src['frames']};records={f:load(Path('output/typed_evidence_next44')/args.key/f'identity_{f}.pt') for f in order}
    texts=records[order[0]]['queries'];out=Path('output/text_video_next63/tracks')/args.key;out.mkdir(parents=True,exist_ok=True)
    model=build_sam3_video_model(checkpoint_path='checkpoints/sam3_modelscope/sam3.pt',load_from_HF=False,strict_state_dict_loading=True,apply_temporal_disambiguation=True,device='cuda',compile=False)
    tracker=model.tracker;tracker.backbone=model.detector.backbone
    with tempfile.TemporaryDirectory(prefix='text_source_track_') as temp,torch.autocast('cuda',dtype=torch.bfloat16):
        width,height=Image.open(paths[order[0]]).size;height=round(height*1008/width);width=1008
        for i,f in enumerate(order):Image.open(paths[f]).convert('RGB').resize((width,height),Image.Resampling.BILINEAR).save(Path(temp)/f'{i:05d}.jpg')
        for qi,text in enumerate(texts):
            destination=out/f'{qi}.pt'
            if destination.exists():continue
            start=time.monotonic();seeds=[]
            for f in order:
                r=records[f]['instances'][qi];admitted=(r['foreground_margin']>0)&((r['masks']>.5).flatten(1).any(1)) if len(r['scores']) else torch.zeros(0,dtype=torch.bool)
                if admitted.any():seeds.append((float(r['scores'][admitted].max()),f,admitted))
            if not seeds:
                torch.save(dict(query=text,frames={},seed_frame=None),destination);continue
            _,seed,admitted=max(seeds,key=lambda r:r[0]);index=order.index(seed)
            state=tracker.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=True)
            masks=records[seed]['instances'][qi]['masks'][admitted]>.5
            for oi,mask in enumerate(masks):tracker.add_new_mask(inference_state=state,frame_idx=index,obj_id=oi+1,mask=mask)
            predictions={}
            for reverse in [False,True]:
                for frame_idx,ids,_low,logits,scores in tracker.propagate_in_video(state,start_frame_idx=index,max_frame_num_to_track=len(order),reverse=reverse,propagate_preflight=True):
                    present=torch.as_tensor(scores).flatten()>0
                    probability=F.interpolate(logits.float().reshape(len(ids),1,*logits.shape[-2:]),size=(h,w),mode='bilinear',align_corners=False)[:,0].sigmoid()
                    union=probability[present].amax(0) if present.any() else torch.zeros(h,w,device=probability.device)
                    predictions[order[int(frame_idx)]]=dict(probability=union.cpu().half(),present=bool(present.any()))
            torch.save(dict(query=text,frames=predictions,seed_frame=seed,seed_instances=len(masks),seconds=time.monotonic()-start),destination)
            del state;torch.cuda.empty_cache();print(args.key,qi,text,'tracked',flush=True)
    (out/'contract.json').write_text(json.dumps(dict(texts=texts,source_frames=order,target_rgb_used=False,seed='highest detector confidence view with positive native identity margin; all admitted instances in that view',extra_online_teacher='SAM3 temporal tracker',source_raster=[h,w]),indent=2)+'\n')
if __name__=='__main__':main()
