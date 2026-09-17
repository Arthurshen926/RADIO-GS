"""Matched source extent refinement: lifted labels versus forward pixel loss.

Both arms use identical groups, views, head initialization, optimizer and steps.
No field, routing, geometry or benchmark supervision is changed.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.scope_union_lifting import lift_mask_union


def balanced_mask_loss(pred,target,valid):
    weights=valid[:,None].float()
    positive=weights*target;negative=weights*(1-target)
    bce=-(positive*pred.clamp_min(1e-6).log()).sum(0)/positive.sum(0).clamp_min(1)
    bce-= (negative*(1-pred).clamp_min(1e-6).log()).sum(0)/negative.sum(0).clamp_min(1)
    dice=1-(2*(pred*positive).sum(0)+1e-6)/((pred*weights).sum(0)+positive.sum(0)+1e-6)
    return (.5*bce+dice).mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    p.add_argument('--residual-dim',type=int,default=0);p.add_argument('--steps',type=int,default=800)
    p.add_argument('--root',type=Path);a=p.parse_args();torch.set_num_threads(4)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'grounded_semantic_transfer96';report=json.loads((parent/'report.json').read_text())
    assert sha256_file(parent/'field.pt')==report['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
    path=root/'caption_region_evidence.pt';assert sha256_file(path)==report['evidence_sha256'];e=torch.load(path,weights_only=False)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and e['geometry_binding']==payload['geometry_binding']
    observed=payload['state_dict']['observed'].to(a.device);groups=report['groups'];h,w=e['masks'].shape[-2:];views=[]
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);ids=[j for j,g in enumerate(groups) if vi in g['views']]
        if not ids:continue
        r=auth['views'][vi];hitpath=root/r['relative_path'];assert sha256_file(hitpath)==r['sha256']
        hit=torch.load(hitpath,map_location=a.device,weights_only=False)
        gid=hit['gaussian_ids'].long();pid=hit['pixel_ids'].long();weight=hit['base_weights'].float()
        full=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight)
        keep=observed[gid];gid,pid,weight=gid[keep],pid[keep],weight[keep]
        mass=torch.zeros_like(full).scatter_add_(0,pid,weight)
        valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8)
        unique,inverse=gid.unique(return_inverse=True)
        mapping=torch.sparse_coo_tensor(torch.stack([pid,inverse]),weight/mass[pid].clamp_min(1e-8),(h*w,len(unique))).coalesce()
        lift_mapping=torch.sparse_coo_tensor(torch.stack([pid,inverse]),weight/full[pid].clamp_min(1e-8),(h*w,len(unique))).coalesce()
        targets=[];lifted=[]
        for j in ids:
            members=[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi]
            targets.append(e['masks'][members].float().amax(0).flatten())
            union,_=lift_mask_union(lift_mapping,e['masks'][members].to(a.device))
            lifted.append(union.cpu())
        views.append(dict(frame=frame,ids=torch.tensor(ids,device=a.device),unique=unique,mapping=mapping,
                          valid=valid,target=torch.stack(targets,1).to(a.device),lifted=torch.stack(lifted,1).to(a.device)))
    for mode in ['lifted','rendered']:
        torch.manual_seed(42)
        architecture=dict(payload['architecture'])
        if a.residual_dim:architecture['grounded_transfer']=dict(architecture['grounded_transfer'],residual_dim=a.residual_dim)
        model=ElementRetrievalField(**architecture).to(a.device);missing=model.load_state_dict(payload['state_dict'],strict=False)
        assert not missing.unexpected_keys and all(k.startswith('grounded_transfer.residual.') for k in missing.missing_keys)
        model.requires_grad_(False);head=model.grounded_transfer;head.decoder.requires_grad_(True)
        if head.residual is not None:head.residual.requires_grad_(True)
        z=head.features(model.latent).detach();optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.001)
        def predict(features,ids):
            raw=F.linear(features,head.decoder.weight[ids],head.decoder.bias[ids])
            if head.residual is not None:
                hidden=head.residual[1](head.residual[0](features))
                raw=raw+F.linear(hidden,head.residual[2].weight[ids],head.residual[2].bias[ids])
            return raw.clamp(1e-6,1-1e-6)
        def evaluate():
            values=[]
            with torch.no_grad():
                for v in views:
                    q=predict(z[v['unique']],v['ids'])
                    pred=torch.sparse.mm(v['mapping'],q)>.5;target=v['target']>.5;valid=v['valid'][:,None]
                    values.extend(((pred&target&valid).sum(0)/((pred|target)&valid).sum(0).clamp_min(1)).tolist())
            return sum(values)/len(values)
        history=[dict(step=0,source_render_iou=evaluate())]
        for step in range(a.steps):
            v=views[step%len(views)];cols=torch.randperm(len(v['ids']),device=a.device)[:8];ids=v['ids'][cols]
            pred=predict(z[v['unique']],ids)
            if mode=='rendered':loss=balanced_mask_loss(torch.sparse.mm(v['mapping'],pred),v['target'][:,cols],v['valid'])
            else:loss=balanced_mask_loss(pred,v['lifted'][:,cols],torch.ones(len(pred),dtype=torch.bool,device=a.device))
            optimizer.zero_grad();loss.backward();optimizer.step()
            if (step+1)%200==0:
                history.append(dict(step=step+1,source_render_iou=evaluate(),last_loss=float(loss.detach())))
                print(a.scene,mode,history[-1],flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        for k,v in payload['state_dict'].items():
            if not k.startswith('grounded_transfer.decoder.'):assert torch.equal(v,state[k]),k
        out=root/('source_extent_'+mode+('_residual' if a.residual_dim else '')+'96');out.mkdir(exist_ok=False)
        torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
        receipt=dict(report,schema='source_extent_objective_control.v1',field_sha256=sha256_file(out/'field.pt'),
            parent_field_sha256=report['field_sha256'],objective=mode,steps=a.steps,residual_dim=a.residual_dim,learning_rate=.001,seed=42,
            checkpoint_selection='fixed final step; no target or held-out label selection',history=history,
            source_only=True,benchmark_data_used=False,all_parent_tensors_unchanged=False,all_parent_field_and_routing_tensors_unchanged=True,
            group_lifting='pixel_union_before_projection',union_implementation_sha256=sha256_file('radio_gs/training/scope_union_lifting.py'),authority_sha256=info['authority_sha256'],implementation_sha256=sha256_file(__file__),
            readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
        (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
