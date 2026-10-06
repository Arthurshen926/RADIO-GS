"""Same fixed point objective, local single/dual-teacher and global controls."""
import argparse,json,time
from pathlib import Path
import torch
from radio_gs.querying.local_element_fusion import make_local_head,local_logits,point_retrieval_loss,FEATURE_NAMES,AUXILIARY_NAMES
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS

ROOT=Path('output/local_element_fusion')

@torch.no_grad()
def scores(logits,row):
    result={}
    for split,ids in SPLITS.items():
        cols=[row['class_ids'].index(c) for c in ids];count=row['counts'][:,cols]
        pred=torch.nn.functional.one_hot(logits[:,cols].argmax(1),len(cols))*row['observed'][:,None]
        tp=(pred*count).sum(0);size=count.sum(0);union=(pred*count.sum(1,keepdim=True)).sum(0)+size-tp
        result[split]=float((tp/union.clamp_min(1))[size>0].mean())
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['local_single','local_dual','global_dual'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);started=time.monotonic();scenes=json.loads((ROOT/'split.json').read_text())['fit']
    rows=[torch.load(ROOT/'training'/f'{n}.pt',map_location='cpu') for n in scenes];dim=16 if a.mode=='local_single' else 20
    total=sum(d['features'].shape[0]*d['features'].shape[1] for d in rows)
    mean=sum(d['features'][:,:,:dim].sum((0,1)) for d in rows)/total
    std=(sum(d['features'][:,:,:dim].square().sum((0,1)) for d in rows)/total-mean.square()).clamp_min(1e-8).sqrt()
    for d in rows:
        d['x']=((d.pop('features')[:,:,:dim]-mean)/std).cuda()
        for key in ['base_logits','average_logits','active','counts','observed']:d[key]=d[key].cuda()
    model=make_local_head(dim).cuda();torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003);mode='global' if a.mode.startswith('global') else 'local';history=[]
    baseline={d['scene']:dict(base=scores(d['base_logits'],d),uniform=scores(d['average_logits'],d)) for d in rows}
    for epoch in range(60):
        optimizer.zero_grad();losses=[]
        for d in rows:
            z=local_logits(model,d['x'],d['base_logits'],d['active'],mode)
            loss=point_retrieval_loss(z,d['counts'],d['observed'])
            regularization=sum(p.square().sum() for p in model.parameters())/sum(p.numel() for p in model.parameters())
            ((loss+.001*regularization)/len(rows)).backward();losses.append(float(loss.detach()))
        optimizer.step();history.append(dict(epoch=epoch,loss=sum(losses)/len(losses)));print(a.mode,epoch,history[-1]['loss'],flush=True)
    with torch.no_grad():result={d['scene']:scores(local_logits(model,d['x'],d['base_logits'],d['active'],mode),d) for d in rows}
    state=dict(network={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=mean,std=std,feature_names=FEATURE_NAMES+(AUXILIARY_NAMES if dim==20 else []),
        dual_teacher=dim==20,mode=mode,fit_scenes=scenes,ground_truth_training=True,validation_labels_read=False,class_ids_as_features=False,
        epochs=60,lr=.003,objective='0.5 balanced point CE + 0.5 soft IoU + 0.001 parameter L2',native_coordinates_frozen=True)
    torch.save(state,ROOT/f'{a.mode}.pt')
    (ROOT/f'{a.mode}_fit.json').write_text(json.dumps(dict(training_only=True,scenes=result,baseline=baseline,history=history,seconds=time.monotonic()-started),indent=2)+'\n')
    print('TRAINING ONLY',a.mode,{sp:sum(d[sp] for d in result.values())/len(result) for sp in SPLITS},flush=True)

if __name__=='__main__':main()
