"""Matched concept extent refinement with equal or contribution-weighted views.

Native field, query readout, routing and independent region heads stay exact.
Source mask union is performed in 2D before linear lifting in both arms.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.training.source_membership_consensus import source_membership_consensus
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene;bindings={}
    def load(name,digest):
        path=root/name;assert sha256_file(path)==digest;bindings[name]=digest
        return torch.load(path,weights_only=False)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    parent=root/'exact_render_query_threshold96';pr=json.loads((parent/'report.json').read_text());payload=load('exact_render_query_threshold96/field.pt',pr['field_sha256'])
    cap=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert cap['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and cap['geometry_binding']==payload['geometry_binding']
    group_path=root/'grounded_semantic_transfer96/report.json';groups=json.loads(group_path.read_text())['groups'];bindings['grounded_semantic_transfer96/report.json']=sha256_file(group_path)
    field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False);head=field.grounded_transfer
    start=head.identity_selection['trusted_concept_start'];assert len(groups)==len(head.prototypes) and head.num_proposals==start+len(groups)
    cache={};n=len(field.latent);pixels=cap['masks'][0].numel()
    with torch.no_grad():
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);js=[j for j,g in enumerate(groups) if vi in g['views']]
            if not js:continue
            rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256']);gid,pid,w=[hit[k].to(a.device) for k in ['gaussian_ids','pixel_ids','base_weights']];gid=gid.long();pid=pid.long();w=w.float()
            pm=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w);weights=w/pm[pid].clamp_min(1e-8)
            mat=torch.sparse_coo_tensor(torch.stack([gid,pid]),weights,(n,pixels)).coalesce();em=torch.zeros(n,device=a.device).scatter_add_(0,gid,weights)
            targets=torch.stack([cap['masks'][[i for i in groups[j]['regions'] if int(cap['proposal_views'][i])==vi]].bool().any(0).flatten() for j in js],1).float().to(a.device)
            lifted=(torch.sparse.mm(mat,targets)/em[:,None].clamp_min(1e-8)).clamp(0,1)
            for col,j in enumerate(js):cache.setdefault(j,[]).append((lifted[:,col].clone(),em))
        z=head.features(field.latent).detach();hidden=head.residual[1](head.residual[0](z)).detach()
        features=torch.cat([z,hidden,torch.ones(n,1,device=a.device)],1)
        initial=torch.cat([head.decoder.weight[start:],head.residual[2].weight[start:],(head.decoder.bias[start:]+head.residual[2].bias[start:])[:,None]],1).detach()
    for mode in ['equal','mass']:
        torch.manual_seed(42);fit={};stats=[]
        with torch.no_grad():
            for j,items in cache.items():
                p=torch.stack([v[0] for v in items]);mass=torch.stack([v[1] for v in items]);seen=mass>1e-6
                if mode=='mass':target,known=source_membership_consensus(p,mass)
                else:target=(p*seen).sum(0)/seen.sum(0).clamp_min(1);known=seen.any(0)
                known &= field.observed;positive=torch.where(known&(target>.5))[0];negative=torch.where(known&(target<=.5))[0]
                if not len(positive) or not len(negative):continue
                fit[j]=(positive,negative,target);stats.append(dict(group=j,positives=len(positive),negatives=len(negative)))
        coefficients=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([coefficients],lr=.001);keys=sorted(fit);history=[]
        for step in range(1200):
            j=keys[step%len(keys)];pos,neg,target=fit[j]
            rows=torch.cat([pos[torch.randint(len(pos),(512,),device=a.device)],neg[torch.randint(len(neg),(512,),device=a.device)]])
            prediction=features[rows]@coefficients[j];loss=(prediction-target[rows]).square().mean()+.001*(coefficients[j]-initial[j]).square().mean()
            opt.zero_grad();loss.backward();opt.step()
            if (step+1)%300==0:history.append(dict(step=step+1,loss=float(loss.detach())));print(a.scene,mode,history[-1],flush=True)
        state={k:v.clone() for k,v in payload['state_dict'].items()};c=coefficients.detach().cpu();d=z.shape[1];r=hidden.shape[1]
        state['grounded_transfer.decoder.weight'][start:]=c[:,:d];state['grounded_transfer.residual.2.weight'][start:]=c[:,d:d+r]
        state['grounded_transfer.decoder.bias'][start:]=c[:,-1]-state['grounded_transfer.residual.2.bias'][start:]
        changed=['grounded_transfer.decoder.weight','grounded_transfer.residual.2.weight','grounded_transfer.decoder.bias']
        for key,v in payload['state_dict'].items():
            if key in changed:assert torch.equal(v[:start],state[key][:start])
            else:assert torch.equal(v,state[key]),key
        out=root/('source_visibility_'+mode+'_extent96');out.mkdir(exist_ok=False);torch.save(dict(payload,state_dict=state),out/'field.pt')
        report=dict(schema='source_visibility_extent_refit.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=pr['field_sha256'],source_only=True,benchmark_data_used=False,
            fusion=mode,union_before_lifting=True,steps=1200,seed=42,learning_rate=.001,coefficient_anchor_penalty=.001,history=history,groups=stats,
            native_field_and_queries_exact=True,independent_region_heads_exact=True,concept_start=start,changed_tensor_keys=changed,
            checkpoint_selection='fixed final step; no validation or benchmark selection',image_prompt_unchanged=True,
            inputs=bindings,authority_sha256=info['authority_sha256'],implementation_sha256=sha256_file(__file__),fusion_implementation_sha256=sha256_file('radio_gs/training/source_membership_consensus.py'))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
