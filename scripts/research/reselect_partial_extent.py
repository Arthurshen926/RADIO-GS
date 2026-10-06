"""Isolate partial-label source validation without refitting next40's field."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.querying.partial_extent_supervision import extent_measures


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    source=Path('output/odds_kernel_extent_next40')/a.key
    data=torch.load(source/'field.pt',map_location='cpu',weights_only=False)
    cache=torch.load(source/'source_training.pt',map_location='cpu',weights_only=False)
    state=data['source_kernel_extent'];q=cache['query'].cuda();gram=torch.exp(state['concentration']*(q@q.T-1))
    base=cache['base_logits'].cuda();prob=base.sigmoid();delta=state['residual'].float().cuda();choices=[]
    for ridge in [.1,1.,10.]:
        inv=torch.linalg.inv(gram+ridge*torch.eye(len(q),device='cuda'))
        prediction=(base+torch.logit((prob+delta@(inv@gram)).clamp(1e-6,1-1e-6))-torch.logit(prob.clamp(1e-6,1-1e-6))).sigmoid()
        metrics={}
        for e in cache['examples']:
            if e['view'] not in cache['held']:continue
            e={k:v.cuda() if torch.is_tensor(v) else v for k,v in e.items()}
            response=torch.sparse.mm(e['op'],prediction[:,e['columns']])
            for k,v in extent_measures(response,e).items():metrics.setdefault(k,[]).extend(v.tolist())
        choices.append(dict(ridge=ridge,**{k:sum(v)/len(v) for k,v in metrics.items()}))
    best=max(choices,key=lambda x:x['known_iou']);old=state['ridge'];state['ridge']=best['ridge']
    state['inverse']=torch.linalg.inv(gram+best['ridge']*torch.eye(len(q),device='cuda')).cpu()
    out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt')
    (out/'selection.json').write_text(json.dumps(dict(key=a.key,old_ridge=old,selected=best,choices=choices,source_only=True,unchanged_training=True,unknown_is_negative=False),indent=2)+'\n')
    print(a.key,old,best,flush=True)


if __name__=='__main__':main()
