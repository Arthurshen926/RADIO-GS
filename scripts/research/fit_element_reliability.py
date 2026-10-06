"""Fixed information x objective experiment; no validation-based selection."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.querying.supervised_scope_reliability import FEATURE_NAMES,NATIVE_FEATURE_NAMES,make_reliability_head
from radio_gs.querying.element_reliability_fit import compiled_probability,retrieval_loss

ROOT=Path('output/element_retrieval_reliability')

def penalty(model):
    return sum(p.square().sum() for p in model.parameters())/sum(p.numel() for p in model.parameters())

def fit_quality(rows,dim,hidden):
    x=torch.cat([d['features'][d['valid_targets']][:,:dim] for d in rows]);y=torch.cat([d['targets'][d['valid_targets']] for d in rows])
    weight=torch.cat([torch.full((int(d['valid_targets'].sum()),),1/int(d['valid_targets'].sum())) for d in rows]);weight/=weight.sum()
    mean=x.mean(0);std=x.std(0).clamp_min(1e-4);x=(x-mean)/std
    model=make_reliability_head(dim,hidden);last=model[-1] if hidden else model
    torch.nn.init.zeros_(last.weight);torch.nn.init.zeros_(last.bias)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();regularization=penalty(model) if hidden else model.weight.square().mean()
        loss=(F.binary_cross_entropy_with_logits(model(x),y,reduction='none').mean(1)*weight).sum()+.001*regularization;loss.backward();return loss
    optimizer.step(closure)
    return model,mean,std

def main():
    p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');p.add_argument('--epochs',type=int,default=60);p.add_argument('--hidden',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);started=time.monotonic();scenes=json.loads((ROOT/'split.json').read_text())['fit']
    rows=[torch.load(ROOT/'training'/f'{n}.pt',map_location='cpu') for n in scenes];dim=14 if a.native else 10
    names=FEATURE_NAMES+(NATIVE_FEATURE_NAMES if a.native else []);tag=('native' if a.native else 'base')+('_mlp' if a.hidden else '')
    model,mean,std=fit_quality(rows,dim,a.hidden)
    def save(mode):
        state=dict(mean=mean.cpu(),std=std.cpu(),feature_names=names,
            fit_scenes=scenes,native_region_identity=a.native,class_ids_as_features=False,ground_truth_training=True,validation_labels_read=False,
            targets=['positive_evidence_weight','negative_evidence_weight'] if mode=='retrieval' else ['mask_precision','visible_category_recall'],
            objective=mode,hidden_dimension=a.hidden,epochs=a.epochs if mode=='retrieval' else 0,optimization='fixed schedule; no validation selection')
        if a.hidden:state['network']={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else:state.update(weight=model.weight.detach().cpu(),bias=model.bias.detach().cpu())
        torch.save(state,ROOT/f'{tag}_{mode}.pt')
    save('quality');model.cuda();mean=mean.cuda();std=std.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.003 if a.hidden else .03)
    fixed=[dict(x=(d['features'][:,:,:dim].cuda()-mean)/std,y=d['targets'].cuda(),active=d['valid_targets'].cuda(),counts=d['counts'].cuda(),observed=d['observed'].cuda()) for d in rows]
    history=[]
    for epoch in range(a.epochs):
        optimizer.zero_grad();metrics=[]
        for data,batch in zip(rows,fixed):
            logits=model(batch['x']);quality=logits.sigmoid();probability=compiled_probability(quality,data)
            element=retrieval_loss(probability,batch['counts'],batch['observed'])
            auxiliary=F.binary_cross_entropy_with_logits(logits[batch['active']],batch['y'][batch['active']])
            regularization=penalty(model) if a.hidden else model.weight.square().mean()
            loss=element+.1*auxiliary+.001*regularization
            (loss/len(rows)).backward();metrics.append(dict(scene=data['scene'],element_loss=float(element.detach()),quality_loss=float(auxiliary.detach())))
        optimizer.step();history.append(dict(epoch=epoch,scenes=metrics))
        print(tag,epoch,sum(r['element_loss'] for r in metrics)/len(rows),flush=True)
    save('retrieval');(ROOT/f'{tag}_fit.json').write_text(json.dumps(dict(training_scenes=scenes,feature_names=names,history=history,
        seconds=time.monotonic()-started,training_labels_used=True,validation_selection=False),indent=2)+'\n')

if __name__=='__main__':main()
