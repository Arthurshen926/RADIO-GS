"""Fit a positive-weight selector from source cross-view mask compatibility.

The candidate's own source view never supplies its quality prior. A supervised
selection episode also excludes its target view from that prior and excludes
same-view candidates. All scene field and extent decoder tensors stay frozen.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file


def cross_view_quality(compatibility,present,origin,excluded=None):
    view_ids=torch.arange(len(present),device=compatibility.device)
    use=present[:,:,None]&(view_ids[:,None,None]!=origin[None,None,:])
    if excluded is not None:use=use&(view_ids[:,None,None]!=excluded)
    return (compatibility*use).sum(0)/use.sum(0).clamp_min(1),use.sum(0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    p.add_argument('--parent',default='source_region_extent96');p.add_argument('--name',default='source_cross_view_ranker96')
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/a.parent;r=json.loads((parent/'report.json').read_text())
    assert sha256_file(parent/'field.pt')==r['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
    epath=root/'caption_region_evidence.pt';assert sha256_file(epath)==r['evidence_sha256'];e=torch.load(epath,weights_only=False)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256'] and e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    assert e['geometry_binding']==payload['geometry_binding']
    architecture=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],proposal_ranker=True))
    field=ElementRetrievalField(**architecture).to(a.device);missing=field.load_state_dict(payload['state_dict'],strict=False)
    assert set(missing.missing_keys)=={'grounded_transfer.proposal_quality','grounded_transfer.proposal_rank_weights'} and not missing.unexpected_keys
    field.requires_grad_(False);head=field.grounded_transfer;groups=r['groups'];proposals=r['proposal_indices']
    origin=(torch.tensor(r['proposal_origins']) if 'proposal_origins' in r else e['proposal_views'][proposals]).to(a.device)
    G=len(groups);R=len(proposals);V=len(info['frames']);h,w=e['masks'].shape[-2:]
    compatibility=torch.zeros(V,G,R,device=a.device);present=torch.zeros(V,G,dtype=torch.bool,device=a.device)
    with torch.no_grad():
        features=head.features(field.latent);response=(head.decoder(features)+head.residual(features)).clamp(1e-6,1-1e-6)
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);record=auth['views'][vi];path=root/record['relative_path'];assert sha256_file(path)==record['sha256']
            hit=torch.load(path,map_location=a.device,weights_only=False);gid=hit['gaussian_ids'].long();pid=hit['pixel_ids'].long();weight=hit['base_weights'].float()
            full=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight);keep=field.observed[gid];gid,pid,weight=gid[keep],pid[keep],weight[keep]
            mass=torch.zeros_like(full).scatter_add_(0,pid,weight);valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8)
            matrix=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,len(field.latent))).coalesce()
            prediction=torch.cat([torch.sparse.mm(matrix,chunk) for chunk in response.split(64,1)],1)>.5
            prediction=(prediction&valid[:,None]).float();area=prediction.sum(0)
            for j,g in enumerate(groups):
                if vi not in g['views']:continue
                ids=[i for i in g['regions'] if int(e['proposal_views'][i])==vi]
                target=e['masks'][ids].bool().any(0).flatten().to(a.device)&valid
                if not target.any():continue
                overlap=target.float()@prediction
                compatibility[vi,j]=overlap/(target.sum()+area-overlap).clamp_min(1);present[vi,j]=True
    # Leave-candidate-origin and leave-episode-target out. Visibility with no
    # detected group supplies no negative label or quality observation.
    def quality(excluded=None):
        return cross_view_quality(compatibility,present,origin,excluded)
    encoder=NativeThreeQueryEncoder(payload['query_encoder'],a.device)
    phrases=[g['phrase'] for g in groups]
    query=encoder.encode_text(phrases);similarity=F.normalize(query,dim=-1)@F.normalize(head.proposal_embeddings,dim=-1).T
    heldout=info['frames'].index(info['train_frames'][-1]);train=[];validation=[]
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);prior,count=quality(vi)
        for j in torch.where(present[vi])[0].tolist():
            eligible=head.proposal_groups[:,j]&(origin!=vi)&(count[j]>0)
            ids=torch.where(eligible)[0]
            if len(ids)<2:continue
            target=compatibility[vi,j,ids]
            if float(target.max())<.1 or float(target.max()-target.min())<.05:continue
            features=torch.stack([similarity[j,ids],prior[j,ids]],1).detach()
            episode=(features,target.detach())
            (validation if vi==heldout else train).append(episode)
    assert train and validation
    raw=torch.tensor([3.,0.],device=a.device,requires_grad=True);optimizer=torch.optim.Adam([raw],lr=.05)
    def measure(episodes,weights):
        return sum(float(y[(x@weights).argmax()]) for x,y in episodes)/len(episodes)
    history=[];best=-1;best_weights=None
    baseline=measure(validation,torch.tensor([1.,0.],device=a.device))
    for step in range(600):
        weights=F.softplus(raw)
        loss=torch.stack([-(F.softmax(y/.1,0)*F.log_softmax(x@weights,0)).sum() for x,y in train]).mean()+1e-4*weights.square().sum()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%100==0:
            weights=F.softplus(raw).detach();score=measure(validation,weights)
            history.append(dict(step=step+1,source_validation_selected_iou=score,weights=weights.cpu().tolist()))
            if score>best:best=score;best_weights=weights.clone()
    with torch.no_grad():head.proposal_quality.copy_(quality()[0]);head.proposal_rank_weights.copy_(best_weights)
    state={k:v.detach().cpu() for k,v in field.state_dict().items()}
    for k,v in payload['state_dict'].items():assert torch.equal(v,state[k]),k
    out=root/a.name;out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
    receipt=dict(r,schema='cross_view_source_ranker.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=r['field_sha256'],
        source_only=True,benchmark_data_used=False,all_parent_tensors_unchanged=True,train_selection_episodes=len(train),validation_selection_episodes=len(validation),
        validation_source_frame=info['train_frames'][-1],validation_baseline_iou=baseline,validation_ranker_iou=best,history=history,
        source_validation_scope='selector supervision held out; base field/extent heads already saw all source views',
        selection='nonnegative native cosine plus source cross-view teacher compatibility; original concept gate unchanged',
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
    (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(a.scene,len(train),len(validation),baseline,best,best_weights.tolist(),flush=True)


if __name__=='__main__':main()
