"""Source-only direct spatial ranking versus the fused semantic carriers."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--base',default='output/dense_language_surface_next43');a=p.parse_args();torch.set_num_threads(4)
    base=Path(a.base);cache=load(Path('output/odds_kernel_extent_next40')/a.key/'source_training.pt');q=F.normalize(cache['query'].cuda(),dim=-1)
    payload=load(base/'gaussian'/a.key/'field.pt');root=Path(payload['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());h,w=auth['metadata']['feature_height'],auth['metadata']['feature_width'];scores={};supports={}
    for mode in ['gaussian','gaussian_gate','surface','surface_robust']:
        state=load(base/mode/a.key/'field.pt')['dense_surface_field'];feat=state['features'];ids=state['gaussian_to_element'].cuda()
        supports[mode]=state['supported'].cuda()[:,None]
        scores[mode]=torch.cat([F.normalize(x.float().cuda(),dim=-1)@q.T for x in feat.split(4096)])[ids]*supports[mode]
    def ap(score,y,known):
        out=[]
        for j in range(score.shape[1]):
            k=known[:,j].bool();target=y[k,j].float();s=score[k,j]
            if not target.any() or target.all():continue
            ranked=target[s.argsort(descending=True)];precision=ranked.cumsum(0)/torch.arange(1,len(ranked)+1,device='cuda')
            out.append(float((precision*ranked).sum()/ranked.sum()))
        return out
    rows=[]
    for e in cache['examples']:
        frame=info['frames'][e['view']];record=load(base/a.key/f'dense_{frame}.pt');rh,rw=record['shape']
        grid=record['features'].float().cuda().transpose(1,2).reshape(1,1536,rh,rw);grid=F.normalize(F.interpolate(grid,size=(h,w),mode='bilinear',align_corners=False)[0].flatten(1).T,dim=-1)
        col=e['columns'].cuda();y=e['target'].cuda();known=e['known'].cuda();op=e['op'].cuda()
        mass={mode:torch.sparse.mm(op,support.float()) for mode,support in supports.items()}
        common=torch.stack([m[:,0]>.5 for m in mass.values()]).all(0)
        known=known*common[:,None]
        row=dict(frame=frame,held=e['view'] in cache['held'],common_supported_pixels=int(common.sum()),direct=ap(grid@q[col].T,y,known))
        for mode,score in scores.items():row[mode]=ap(torch.sparse.mm(op,score[:,col])/mass[mode].clamp_min(1e-8),y,known)
        rows.append(row)
    report=dict(key=a.key,metric='AP within source known positive/negative pixels; not full-object benchmark',rows=rows)
    report['macro']={split:{mode:sum(v for r in rows if r['held']==held for v in r[mode])/sum(len(r[mode]) for r in rows if r['held']==held) for mode in ['direct']+list(scores)} for split,held in [('fit',False),('held',True)]}
    (base/a.key/'source_readback.json').write_text(json.dumps(report,indent=2)+'\n');print(report['macro'])


if __name__=='__main__':main()
