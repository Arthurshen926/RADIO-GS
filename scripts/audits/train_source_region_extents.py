"""Preserve region alternatives instead of collapsing masks into concept unions."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.train_rendered_extent import balanced_mask_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    p.add_argument('--root',type=Path);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'grounded_semantic_transfer96';report=json.loads((parent/'report.json').read_text())
    assert sha256_file(parent/'field.pt')==report['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
    path=root/'caption_region_evidence.pt';assert sha256_file(path)==report['evidence_sha256'];e=torch.load(path,weights_only=False)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert e['geometry_binding']==payload['geometry_binding'] and e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    observed=payload['state_dict']['observed'].to(a.device);groups=report['groups']
    candidates=sorted({i for g in groups for i in g['regions']});admitted=[]
    for i in candidates:
        vi=int(e['proposal_views'][i]);known=e['seen'][vi]&observed.cpu();target=e['region_probability'][i].float()
        if int((known&(target>.5)).sum())>=8 and int((known&(target<.1)).sum())>=8:admitted.append(i)
    architecture=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],num_proposals=len(admitted),residual_dim=128))
    model=ElementRetrievalField(**architecture).to(a.device)
    keep={k:v for k,v in payload['state_dict'].items() if not k.startswith('grounded_transfer.decoder.')}
    missing=model.load_state_dict(keep,strict=False);assert not missing.unexpected_keys and all(k.startswith('grounded_transfer.') for k in missing.missing_keys)
    model.requires_grad_(False);head=model.grounded_transfer;head.decoder.requires_grad_(True);head.residual.requires_grad_(True)
    z=head.features(model.latent).detach()
    with torch.no_grad():
        head.proposal_embeddings.copy_(e['native_descriptors'][admitted].float().to(a.device))
        for j,g in enumerate(groups):head.proposal_groups[:,j]=torch.tensor([i in g['regions'] for i in admitted],device=a.device)
        for ri,i in enumerate(admitted):
            vi=int(e['proposal_views'][i]);target=e['region_probability'][i].float().to(a.device);known=e['seen'][vi].to(a.device)&observed
            pos=torch.where(known&(target>.5))[0];neg=torch.where(known&(target<.1))[0]
            rows=torch.cat([pos[torch.randint(len(pos),(2048,),device=a.device)],neg[torch.randint(len(neg),(2048,),device=a.device)]])
            x=torch.cat([z[rows],torch.ones(len(rows),1,device=a.device)],1)
            ridge=torch.eye(x.shape[1],device=a.device)*.001;ridge[-1,-1]=0
            coefficients=torch.linalg.solve(x.T@x/len(x)+ridge,x.T@target[rows]/len(x))
            head.decoder.weight[ri].copy_(coefficients[:-1]);head.decoder.bias[ri].copy_(coefficients[-1])
    h,w=e['masks'].shape[-2:];views=[]
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);ids=[j for j,i in enumerate(admitted) if int(e['proposal_views'][i])==vi]
        if not ids:continue
        r=auth['views'][vi];hitpath=root/r['relative_path'];assert sha256_file(hitpath)==r['sha256']
        hit=torch.load(hitpath,map_location=a.device,weights_only=False);gid=hit['gaussian_ids'].long();pid=hit['pixel_ids'].long();weight=hit['base_weights'].float()
        full=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight);use=observed[gid];gid,pid,weight=gid[use],pid[use],weight[use]
        mass=torch.zeros_like(full).scatter_add_(0,pid,weight);valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8)
        unique,inverse=gid.unique(return_inverse=True);mapping=torch.sparse_coo_tensor(torch.stack([pid,inverse]),weight/mass[pid].clamp_min(1e-8),(h*w,len(unique))).coalesce()
        target=e['masks'][[admitted[j] for j in ids]].float().flatten(1).T.to(a.device)
        views.append(dict(ids=torch.tensor(ids,device=a.device),unique=unique,mapping=mapping,target=target,valid=valid))
    def predict(v,ids):
        features=z[v['unique']];hidden=head.residual[1](head.residual[0](features))
        return (F.linear(features,head.decoder.weight[ids],head.decoder.bias[ids])+F.linear(hidden,head.residual[2].weight[ids],head.residual[2].bias[ids])).clamp(1e-6,1-1e-6)
    def score():
        values=[]
        with torch.no_grad():
            for v in views:
                for start in range(0,len(v['ids']),16):
                    ids=v['ids'][start:start+16];pred=torch.sparse.mm(v['mapping'],predict(v,ids))>.5
                    target=v['target'][:,start:start+16]>.5;valid=v['valid'][:,None]
                    values.extend(((pred&target&valid).sum(0)/((pred|target)&valid).sum(0).clamp_min(1)).tolist())
        return sum(values)/len(values)
    history=[dict(step=0,source_render_iou=score())];optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.001)
    for step in range(800):
        v=views[step%len(views)];cols=torch.randperm(len(v['ids']),device=a.device)[:8]
        pred=torch.sparse.mm(v['mapping'],predict(v,v['ids'][cols]));loss=balanced_mask_loss(pred,v['target'][:,cols],v['valid'])
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%200==0:
            history.append(dict(step=step+1,source_render_iou=score(),last_loss=float(loss.detach())));print(a.scene,history[-1],flush=True)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    for k,v in keep.items():assert torch.equal(v,state[k]),k
    out=root/'source_region_extent96';out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
    r=dict(report,schema='source_region_extent.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=report['field_sha256'],
           all_parent_tensors_unchanged=False,all_parent_field_and_routing_tensors_unchanged=True,source_only=True,benchmark_data_used=False,
           proposal_indices=admitted,proposal_count=len(admitted),steps=800,seed=42,learning_rate=.001,history=history,
           selection='original source concept gate, then highest native image/text cosine among associated region proposals',
           checkpoint_selection='fixed final step',implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
