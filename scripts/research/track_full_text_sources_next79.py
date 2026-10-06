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
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='checkpoints/sam3_modelscope/sam3.pt');parser.add_argument('--scene-input');parser.add_argument('--queries');parser.add_argument('--seed-manifest');parser.add_argument('--key',required=True);parser.add_argument('--raster-scale',type=int,default=1);parser.add_argument('--low-output-base');parser.add_argument('--seed-base');parser.add_argument('--output-base',default='output/full_text_tracks_next79');parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shards',type=int,default=1);parser.add_argument('--query-indices',type=int,nargs='*');args=parser.parse_args();torch.set_num_threads(4)
    if args.scene_input:
        scene=json.loads(Path(args.scene_input).read_text());texts=[q['text'] for q in json.loads(Path(args.queries).read_text())['queries']]
        authority=json.loads(Path(scene['source_rgb_authority']).read_text());paths={int(r['image_id'].split('_')[-1]):Path(r['path']) for r in authority['images']}
        order=scene['source_frames'];h,w=96,128;old_order=[];records={}
        seed_manifest=json.loads(Path(args.seed_manifest).read_text())['seeds']
        out=Path(args.output_base)/args.key;out.mkdir(parents=True,exist_ok=True)
    else:
        data=load(Path('output/native_rebuild_next54/mixture_masscal')/args.key/'field.pt');root=Path(data['root'])
        info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width']
        old_order=sorted(info['train_frames']);records={f:load(Path('output/typed_evidence_next44')/args.key/f'identity_{f}.pt') for f in old_order}
        source_root=Path('output/source_coverage_next72')/args.key.removeprefix('lerf_');authority=json.loads((source_root/'all.json').read_text());paths={int(r['image_id'].split('_')[-1]):Path(r['path']) for r in authority['images']};order=json.loads((source_root/'cohorts.json').read_text())['frames']['all']
        texts=records[old_order[0]]['queries'];out=Path(args.output_base)/args.key;out.mkdir(parents=True,exist_ok=True)
    model=build_sam3_video_model(checkpoint_path=args.checkpoint,load_from_HF=False,strict_state_dict_loading=True,apply_temporal_disambiguation=True,device='cuda',compile=False)
    tracker=model.tracker;tracker.backbone=model.detector.backbone
    with tempfile.TemporaryDirectory(prefix='text_source_track_') as temp,torch.autocast('cuda',dtype=torch.bfloat16):
        width,height=Image.open(paths[order[0]]).size;height=round(height*1008/width);width=1008
        for i,f in enumerate(order):Image.open(paths[f]).convert('RGB').resize((width,height),Image.Resampling.BILINEAR).save(Path(temp)/f'{i:05d}.jpg')
        for qi,text in enumerate(texts):
            if qi%args.shards!=args.shard_index:continue
            if args.query_indices is not None and qi not in args.query_indices:continue
            destination=out/f'{qi}.pt'
            if destination.exists() and not args.scene_input:continue
            start=time.monotonic();seeds=[]
            for f in old_order:
                r=records[f]['instances'][qi];admitted=(r['foreground_margin']>0)&((r['masks']>.5).flatten(1).any(1)) if len(r['scores']) else torch.zeros(0,dtype=torch.bool)
                if admitted.any():seeds.append((float(r['scores'][admitted].max()),f,admitted))
            external=load(seed_manifest[str(qi)]) if args.scene_input and str(qi) in seed_manifest else None
            if not args.scene_input:external=load(Path(args.seed_base)/args.key/f'{qi}.pt') if args.seed_base else None
            if not seeds and external is None:
                torch.save(dict(query=text,frames={},seed_frame=None),destination);continue
            if external is not None:seed=external['seed_frame'];masks=external['masks']
            else:
                _,seed,admitted=max(seeds,key=lambda r:r[0]);masks=records[seed]['instances'][qi]['masks'][admitted]>.5
            index=order.index(seed)
            state=tracker.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=True)
            for oi,mask in enumerate(masks):tracker.add_new_mask(inference_state=state,frame_idx=index,obj_id=oi+1,mask=mask)
            predictions={};low_predictions={}
            for reverse in [False,True]:
                for frame_idx,ids,_low,logits,scores in tracker.propagate_in_video(state,start_frame_idx=index,max_frame_num_to_track=len(order),reverse=reverse,propagate_preflight=True):
                    present=torch.as_tensor(scores).flatten()>0
                    probability=F.interpolate(logits.float().reshape(len(ids),1,*logits.shape[-2:]),size=(h*args.raster_scale,w*args.raster_scale),mode='bilinear',align_corners=False)[:,0].sigmoid()
                    union=probability[present].amax(0) if present.any() else torch.zeros(h*args.raster_scale,w*args.raster_scale,device=probability.device)
                    if args.low_output_base:
                        low=F.interpolate(logits.float().reshape(len(ids),1,*logits.shape[-2:]),size=(h,w),mode='bilinear',align_corners=False)[:,0].sigmoid()
                        low_predictions[order[int(frame_idx)]]=dict(probability=(low[present].amax(0) if present.any() else torch.zeros(h,w,device=low.device)).cpu().half(),present=bool(present.any()))
                    predictions[order[int(frame_idx)]]=dict(probability=union.cpu().half(),present=bool(present.any()))
            torch.save(dict(query=text,frames=predictions,seed_frame=seed,seed_instances=len(masks),seed_evidence=external,seconds=time.monotonic()-start),destination)
            if args.low_output_base:
                low_out=Path(args.low_output_base)/args.key;low_out.mkdir(parents=True,exist_ok=True)
                torch.save(dict(query=text,frames=low_predictions,seed_frame=seed,seed_instances=len(masks),seed_evidence=external,same_tracking_run_as=str(destination)),low_out/f'{qi}.pt')
            del state;torch.cuda.empty_cache();print(args.key,qi,text,'tracked',flush=True)
    (out/('contract_'+('_'.join(map(str,args.query_indices)) if args.query_indices is not None else f'all_{args.shard_index}_of_{args.shards}')+'.json')).write_text(json.dumps(dict(texts=texts,source_frames=order,seed_candidates=args.seed_manifest if args.scene_input else old_order if not args.seed_base else args.seed_base,query_indices=args.query_indices,shard_index=args.shard_index,shards=args.shards,target_rgb_used=False,seed='highest detector confidence view with positive native identity margin; all admitted instances in that view',extra_online_teacher='SAM3 temporal tracker',source_raster=[h*args.raster_scale,w*args.raster_scale],low_output_base=args.low_output_base),indent=2)+'\n')
if __name__=='__main__':main()
