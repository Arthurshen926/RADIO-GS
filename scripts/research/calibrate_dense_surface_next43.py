"""Calibrate the actual dense/fallback mixture, without re-encoding features."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.source_kernel_extent import attach_kernel_native


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
    cache=load(Path('output/odds_kernel_extent_next40')/a.key/'source_training.pt');q=F.normalize(cache['query'].cuda(),dim=-1);fallback=cache['base_logits'].cuda().sigmoid()
    first=load(Path(a.source)/'gaussian'/a.key/'field.pt');parent=load(first['base_field']);model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.eval().requires_grad_(False);attach_kernel_native(model,first)
    with torch.no_grad():threshold=model.native_query_thresholds['text'](q)
    examples=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in e.items()} for e in cache['examples'] if e['view'] not in cache['held']]
    for mode in ['gaussian','gaussian_gate','surface','surface_robust']:
        data=load(Path(a.source)/mode/a.key/'field.pt');s=data['dense_surface_field'];ids=s['gaussian_to_element'].cuda();supported=s['supported'].cuda()
        with torch.no_grad():cosine=torch.cat([F.normalize(part.float().cuda(),dim=-1)@q.T for part in s['features'].split(4096)])[ids]
        scale=torch.nn.Parameter(torch.tensor(3.,device='cuda'));offset=torch.nn.Parameter(torch.tensor(0.,device='cuda'));opt=torch.optim.Adam([scale,offset],lr=.02)
        for step in range(100):
            e=examples[step%len(examples)];col=e['columns'];value=(scale.exp()*(cosine[:,col]-threshold[col]+offset)).sigmoid()
            value=torch.where(supported[:,None],value,fallback[:,col]);pred=torch.sparse.mm(e['op'],value);y=e['target'];pos=e['known']*y;neg=e['known']*(1-y)
            loss=(((pred-y).square()*pos).sum()/pos.sum().clamp_min(1)+((pred-y).square()*neg).sum()/neg.sum().clamp_min(1))/2
            opt.zero_grad();loss.backward();opt.step()
        s['scale']=float(scale.exp());s['offset']=float(offset);out=Path(a.output)/mode/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt')
        report=json.loads((Path(a.source)/mode/a.key/'construction.json').read_text());report.update(scale=s['scale'],offset=s['offset'],fallback_in_calibration=True,source_only=True)
        (out/'construction.json').write_text(json.dumps(report,indent=2)+'\n');print(a.key,mode,s['scale'],s['offset'],flush=True)


if __name__=='__main__':main()
