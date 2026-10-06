"""Independent foreground/context identity evidence for online detections.

Crop embeddings are used only to accept identity proposals, never as element
membership labels. Each decision depends on the single runtime query.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.querying.query_region_evidence import same_window
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime


def load(path):return torch.load(path,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--evidence-base',default='output/typed_evidence_next44');p.add_argument('--source-authority');p.add_argument('--query-encodings');a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
    base=Path('output/taskwise_scope_composition_next38')/a.key;data=load(base/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());src=json.loads((root/'source_frames.json').read_text());frames={r['frame_idx']:r for r in src['frames']}
    if a.key.startswith('scannet_'):
        directory=root.parent.parent/('evaluation' if 'expanded_budget' in str(root) else 'evaluation_complete_next28');encoding=load(directory/'queries.pt')
    else:encoding=load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]
    if a.query_encodings:encoding=load(a.query_encodings)
    q=F.normalize(encoding['encoding'].float().cuda(),dim=-1)
    paths={f:Path(src['image_dir'])/frames[f]['source_file'] for f in info['train_frames']}
    if a.source_authority:paths={int(r['image_id'].split('_')[-1]):Path(r['path']) for r in json.loads(Path(a.source_authority).read_text())['images']}
    teacher=load(root/info['teacher_files']['siglip']['path']);runtime=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device('cuda'))
    out=Path(a.evidence_base)/a.key;kept=total=0
    for frame,path in paths.items():
        detection=load(out/f'detection_{frame}.pt')
        if detection['queries']!=encoding['texts']:raise ValueError('query encoding order differs from detections')
        rgb=torch.from_numpy(np.array(Image.open(path).convert('RGB'),copy=True)).permute(2,0,1).float()/255
        candidates=[];positions=[]
        for qi,row in enumerate(detection['instances']):
            row['foreground_margin']=torch.full((len(row['scores']),),-1.)
            for j,m in enumerate(row['masks']):
                if not (m>.5).any():continue
                mask=F.interpolate((m>.5)[None,None].float(),size=rgb.shape[1:],mode='nearest-exact')[0,0].bool();triple,_=same_window(rgb,mask);candidates.append(triple);positions.append((qi,j))
        if candidates:
            batch=torch.cat(candidates);feature=torch.cat([runtime.encode(x).float() for x in batch.split(8)]).reshape(-1,3,q.shape[-1]);feature=F.normalize(feature,dim=-1)
            for (qi,j),f in zip(positions,feature):detection['instances'][qi]['foreground_margin'][j]=(f[1]@q[qi]-f[2]@q[qi]).cpu()
        for r in detection['instances']:kept+=int((r['foreground_margin']>0).sum());total+=len(r['scores'])
        torch.save(detection,out/f'identity_{frame}.pt');print(a.key,frame,kept,total,flush=True)
    (out/'identity.json').write_text(json.dumps(dict(key=a.key,accepted=kept,proposals=total,seconds=time.monotonic()-start,source_frames=list(paths),source_authority=a.source_authority,criterion='native foreground cosine greater than same-window context cosine for the same query',query_batch_competition=False,element_feature_writes=False,not_verified_identity=True),indent=2)+'\n')

if __name__=='__main__':main()
