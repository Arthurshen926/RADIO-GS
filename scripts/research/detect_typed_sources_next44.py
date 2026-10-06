"""ONLINE text-set grounding on authorized source RGB, never target RGB.

Query texts are runtime inputs, not map-training vocabulary. Outputs are
transient query evidence; this has an additional online VFM/input-access budget.
"""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model,sam3_autocast_context


def load(path):return torch.load(path,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--output-base',default='output/typed_evidence_next44');p.add_argument('--raster-scale',type=int,default=1);p.add_argument('--source-authority');p.add_argument('--queries-json',help='Runtime JSON list of verbatim queries; independent of benchmark cache');p.add_argument('--native-resolution',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    start=time.monotonic();base=Path('output/taskwise_scope_composition_next38')/a.key
    data=load(base/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());source=json.loads((root/'source_frames.json').read_text())
    queries=json.loads(Path(a.queries_json).read_text()) if a.queries_json else load(base/'anchored/pipeline_calibrated/scores.pt')['texts']
    h,w=auth['metadata']['feature_height']*a.raster_scale,auth['metadata']['feature_width']*a.raster_scale;frames={r['frame_idx']:r for r in source['frames']}
    paths={f:Path(source['image_dir'])/frames[f]['source_file'] for f in info['train_frames']}
    if a.source_authority:
        authority=json.loads(Path(a.source_authority).read_text());paths={int(r['image_id'].split('_')[-1]):Path(r['path']) for r in authority['images']}
    processor=_load_sam3_model(checkpoint_path='checkpoints/sam3_modelscope/sam3.pt',device='cuda',confidence_threshold=.5,dtype='float32',resolution=1008)
    out=Path(a.output_base)/a.key;out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for frame,path in paths.items():
        destination=out/f'detection_{frame}.pt'
        if destination.exists():
            cached=load(destination)
            if cached['queries']!=queries or cached.get('native_resolution',False)!=a.native_resolution:raise ValueError('Use a separate output directory for a different query or raster contract')
            continue
        rgb=Image.open(path).convert('RGB')
        if a.native_resolution:w,h=rgb.size
        with sam3_autocast_context('cuda',torch.bfloat16):state=processor.set_image(rgb)
        rows=[]
        for qi,query in enumerate(queries):
            with sam3_autocast_context('cuda',torch.bfloat16):result=processor.set_text_prompt(query,dict(state))
            probabilities=result['masks_logits'];scores=result['scores'];count=len(scores)
            if count:
                masks=F.interpolate(probabilities.float().reshape(count,1,*probabilities.shape[-2:]),size=(h,w),mode='bilinear',align_corners=False)[:,0]
            else:masks=torch.empty(0,h,w,device='cuda')
            rows.append(dict(masks=masks.half().cpu(),scores=scores.float().cpu()))
        torch.save(dict(frame=frame,queries=queries,instances=rows,native_resolution=a.native_resolution,raster_hw=[h,w],source_rgb=str(path)),destination)
        reports.append(dict(frame=frame,instances=sum(len(r['scores']) for r in rows),rgb=str(path)))
        print(a.key,frame,reports[-1]['instances'],f'{time.monotonic()-start:.1f}s',flush=True)
    (out/'detection.json').write_text(json.dumps(dict(key=a.key,queries=queries,source_frames=list(paths),native_resolution=a.native_resolution,source_authority=a.source_authority,seconds=time.monotonic()-start,task='text_union',online_query_conditioned=True,map_training=False,target_rgb_used=False,confidence=.5,records=reports),indent=2)+'\n')

if __name__=='__main__':main()
