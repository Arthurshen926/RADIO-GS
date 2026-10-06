"""Fit category competition on the same source-caption positive/negative sets."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--positive-contract',choices=['any','all'],default='any');p.add_argument('--source',default='output/pipeline_kernel_extent_next40');p.add_argument('--output');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    source=Path(a.source)/a.key;data=load(source/'field.pt');cache=load(source/'source_training.pt');s=data['source_kernel_extent']
    q=F.normalize(s['anchors'].float().cuda(),dim=1);kernel=torch.exp(s['concentration']*(q@q.T-1));coefficient=s['inverse'].cuda()@kernel
    baseline=cache['base_logits'].cuda().sigmoid() if s.get('readout')=='odds' else cache['baseline'].float().cuda()
    probability=(baseline+s['residual'].float().cuda()@coefficient).clamp(1e-6,1-1e-6);logits=torch.logit(probability)
    if s.get('readout')=='odds':logits=cache['base_logits'].cuda()+(logits-torch.logit(baseline.clamp(1e-6,1-1e-6)))
    sets={'train':[],'validation':[]}
    with torch.no_grad():
        for e in cache['examples']:
            columns=e['columns'].cuda();score=torch.sparse.mm(e['op'].cuda(),logits[:,columns]);positive=(e['target'].cuda()>.5)&e['known'].cuda().bool();negative=(e['target'].cuda()<.5)&e['known'].cuda().bool()
            valid=positive.any(1)&negative.any(1);ids=torch.where(valid)[0]
            if len(ids)>2048:ids=ids[torch.linspace(0,len(ids)-1,2048,device='cuda').long()]
            if len(ids):sets['validation' if e['view'] in cache['held'] else 'train'].append((score[ids],positive[ids],negative[ids],columns))
    direction=torch.nn.Parameter(torch.zeros(q.shape[1],device='cuda'));opt=torch.optim.Adam([direction],lr=.01)
    def loss(example):
        score,pos,neg,columns=example;score=score+(q[columns]@direction).tanh()[None]
        if a.positive_contract=='all':
            # These source statements are jointly positive, not an ambiguous
            # choice where correctly ranking just one alias satisfies the loss.
            pair=F.softplus(score[:,None,:]-score[:,:,None])
            pair=(pair*neg[:,None,:]).sum(2)/neg.sum(1).clamp_min(1)[:,None]
            weight=pos/pos.sum(0).clamp_min(1)[None]
            return (pair*weight).sum()/weight.sum().clamp_min(1e-8)
        total=score.masked_fill(~(pos|neg),-1e4).logsumexp(1)-score.masked_fill(~pos,-1e4).logsumexp(1)
        # Balance observed positive concepts instead of background pixel count.
        inverse_frequency=1/pos.sum(0).clamp_min(1)
        weight=(pos*inverse_frequency).sum(1)/pos.sum(1).clamp_min(1)
        return (total*weight).sum()/weight.sum().clamp_min(1e-8)
    @torch.no_grad()
    def measure():return {name:sum(float(loss(e)) for e in examples)/len(examples) for name,examples in sets.items()}
    history=[dict(step=0,loss=measure())]
    for step in range(400):
        value=loss(sets['train'][step%len(sets['train'])])+.1*(q@direction).tanh().square().mean()
        opt.zero_grad();value.backward();opt.step()
        if (step+1)%100==0:row=dict(step=step+1,loss=measure());history.append(row);print(a.key,row,flush=True)
    data['kernel_category_competition']=dict(direction=direction.detach().cpu())
    out=Path(a.output or ('output/competitive_kernel_extent_next40' if a.positive_contract=='any' else 'output/all_positive_kernel_extent_next40'))/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt')
    (out/'training.json').write_text(json.dumps(dict(key=a.key,history=history,source_queries=len(q),source_only=True,unknown_excluded=True,positive_contract=a.positive_contract,benchmark_vocabulary_used=False),indent=2)+'\n')


if __name__=='__main__':main()
