"""Matched rendered supervision: full pseudo negatives vs cross-view abstention.

Warm starts the retained rendered head, preserves native/independent branches,
and changes only source pixel negative admission. No benchmark labels are read.
"""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def masked_loss(pred,target,known):
    weight=known.float();pos=weight*target;neg=weight*(1-target)
    bce=-(pos*pred.clamp_min(1e-6).log()).sum(0)/pos.sum(0).clamp_min(1)
    bce-=(neg*(1-pred).clamp_min(1e-6).log()).sum(0)/neg.sum(0).clamp_min(1)
    dice=1-(2*(pred*pos).sum(0)+1e-6)/((pred*weight).sum(0)+pos.sum(0)+1e-6)
    return (.5*bce+dice).mean()


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene;bindings={}
    def load(name,digest):
        path=root/name;assert sha256_file(path)==digest;bindings[name]=digest;return torch.load(path,weights_only=False)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    pp=root/'exact_render_query_threshold96';pr=json.loads((pp/'report.json').read_text());payload=load('exact_render_query_threshold96/field.pt',pr['field_sha256'])
    e=load('caption_region_evidence.pt',json.loads((root/'caption_region_evidence.json').read_text())['sha256']);assert e['geometry_binding']==payload['geometry_binding'] and e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    gp=root/'grounded_semantic_transfer96/report.json';groups=json.loads(gp.read_text())['groups'];bindings['grounded_semantic_transfer96/report.json']=sha256_file(gp)
    field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False);h=field.grounded_transfer;start=h.identity_selection['trusted_concept_start'];G=len(groups)
    assert h.residual[0].out_features==256 and not h.residual[2].weight[:start,128:].any() and not h.residual[2].weight[start:,:128].any()
    z=h.features(field.latent).detach();N=len(z);P=e['masks'][0].numel();views=[];total=torch.zeros(N,G,device=a.device);mass=torch.zeros_like(total)
    with torch.no_grad():
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);ids=[j for j,g in enumerate(groups) if vi in g['views']]
            if not ids:continue
            rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256']);gid,pid,w=[hit[k].to(a.device) for k in ['gaussian_ids','pixel_ids','base_weights']];gid=gid.long();pid=pid.long();w=w.float()
            full=torch.zeros(P,device=a.device).scatter_add_(0,pid,w);keep=field.observed[gid];gid,pid,w=gid[keep],pid[keep],w[keep];pm=torch.zeros_like(full).scatter_add_(0,pid,w)
            valid=(pm/full.clamp_min(1e-8)>.5)&(full>1e-8);unique,inverse=gid.unique(return_inverse=True);weight=w/pm[pid].clamp_min(1e-8)
            mat=torch.sparse_coo_tensor(torch.stack([pid,inverse]),weight,(P,len(unique))).coalesce();em=torch.zeros(len(unique),device=a.device).scatter_add_(0,inverse,weight)
            target=torch.stack([e['masks'][[i for i in groups[j]['regions'] if int(e['proposal_views'][i])==vi]].bool().any(0).flatten() for j in ids],1).float().to(a.device)
            num=torch.sparse.mm(mat.transpose(0,1),target);ids=torch.tensor(ids,device=a.device)
            total[unique[:,None],ids[None]]+=num;mass[unique[:,None],ids[None]]+=em[:,None]
            views.append(dict(frame=frame,ids=ids,rows=unique,mat=mat,target=target,valid=valid,num=num,em=em))
        stats=[]
        for v in views:
            rows,ids=v['rows'],v['ids'];den=(mass[rows[:,None],ids[None]]-v['em'][:,None]).clamp_min(0);num=(total[rows[:,None],ids[None]]-v['num']).clamp_min(0)
            support=torch.sparse.mm(v['mat'],(den>1e-6).float());other=torch.sparse.mm(v['mat'],torch.where(den>1e-6,num/den.clamp_min(1e-8),0))/support.clamp_min(1e-8)
            v['unknown']=(v['target']<.5)&(other>.5)&(support>.5)&v['valid'][:,None]
            v['initial']=torch.sparse.mm(v['mat'],(F.linear(z[rows],h.decoder.weight[start:][ids],h.decoder.bias[start:][ids])+F.linear(F.relu(F.linear(z[rows],h.residual[0].weight[128:],h.residual[0].bias[128:])),h.residual[2].weight[start:,128:][ids],h.residual[2].bias[start:][ids])).clamp(1e-6,1-1e-6))
            stats.append(dict(frame=v['frame'],groups=len(ids),negative_pixels=int(((v['target']<.5)&v['valid'][:,None]).sum()),conflicting_negative_pixels=int(v['unknown'].sum())))
    del total,mass
    for mode in ['full','abstain']:
        torch.manual_seed(42);linear=nn.Linear(96,G).to(a.device);hidden=nn.Linear(96,128).to(a.device);output=nn.Linear(128,G).to(a.device)
        with torch.no_grad():
            linear.weight.copy_(h.decoder.weight[start:]);linear.bias.copy_(h.decoder.bias[start:]);hidden.weight.copy_(h.residual[0].weight[128:]);hidden.bias.copy_(h.residual[0].bias[128:]);output.weight.copy_(h.residual[2].weight[start:,128:]);output.bias.copy_(h.residual[2].bias[start:])
        opt=torch.optim.Adam(list(linear.parameters())+list(hidden.parameters())+list(output.parameters()),lr=.001);history=[]
        for step in range(600):
            v=views[step%len(views)];cols=torch.randperm(len(v['ids']),device=a.device)[:8];ids=v['ids'][cols];features=z[v['rows']]
            element=(F.linear(features,linear.weight[ids],linear.bias[ids])+F.linear(F.relu(hidden(features)),output.weight[ids],output.bias[ids])).clamp(1e-6,1-1e-6)
            pred=torch.sparse.mm(v['mat'],element);truth=v['target'][:,cols];known=v['valid'][:,None].expand_as(pred)
            if mode=='abstain':known=known&~v['unknown'][:,cols]
            loss=masked_loss(pred,truth,known)
            if mode=='abstain':
                unknown=v['unknown'][:,cols];loss=loss+((pred-v['initial'][:,cols]).square()*unknown).sum()/unknown.sum().clamp_min(1)
            opt.zero_grad();loss.backward();opt.step()
            if (step+1)%200==0:history.append(dict(step=step+1,loss=float(loss.detach())));print(a.scene,mode,history[-1],flush=True)
        state={k:v.clone() for k,v in payload['state_dict'].items()}
        state['grounded_transfer.decoder.weight'][start:]=linear.weight.detach().cpu();state['grounded_transfer.decoder.bias'][start:]=linear.bias.detach().cpu()
        state['grounded_transfer.residual.0.weight'][128:]=hidden.weight.detach().cpu();state['grounded_transfer.residual.0.bias'][128:]=hidden.bias.detach().cpu()
        state['grounded_transfer.residual.2.weight'][start:,128:]=output.weight.detach().cpu();state['grounded_transfer.residual.2.bias'][start:]=output.bias.detach().cpu()
        for key,value in payload['state_dict'].items():
            if key.startswith('grounded_transfer.decoder.') or key.startswith('grounded_transfer.residual.2.'):assert torch.equal(value[:start],state[key][:start])
            elif key.startswith('grounded_transfer.residual.0.'):assert torch.equal(value[:128],state[key][:128])
            else:assert torch.equal(value,state[key]),key
        out=root/('source_rendered_partial_'+mode+'96');out.mkdir(exist_ok=False);torch.save(dict(payload,state_dict=state),out/'field.pt')
        report=dict(schema='rendered_partial_extent_refinement.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=pr['field_sha256'],policy=mode,steps=600,learning_rate=.001,seed=42,source_only=True,benchmark_data_used=False,union_before_lifting=True,source_views=stats,history=history,native_query_state_exact=True,independent_region_heads_exact=True,concept_start=start,checkpoint_selection='fixed final step',unknown_preservation_weight=1. if mode=='abstain' else 0.,inputs=bindings,authority_sha256=info['authority_sha256'],implementation_sha256=sha256_file(__file__))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
