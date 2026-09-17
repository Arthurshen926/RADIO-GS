"""Frozen-field automatic-seed membership control; source-only joint 2D/3D fit."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import ElementRetrievalField, load_element_retrieval_field
from radio_gs.training.correspondence_audit import training_query_conflicts
from radio_gs.training.reliable_correspondence import source_region_query_tasks
from radio_gs.training.retrieval_supervision import episode_membership_targets, membership_loss, validate_episode_split
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser()
    for name in ('root','geometry'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('parent','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--semantic-only',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(777)
    root=a.root;out=root/a.output;out.mkdir(exist_ok=False)
    implementation={str(path):sha256_file(path) for path in [Path(__file__),Path('radio_gs/models/element_retrieval_field.py'),Path('radio_gs/models/structural_membership_readout.py')]}
    info=json.loads((root/'teachers.json').read_text());receipt=json.loads((root/'episodes.json').read_text())
    assert sha256_file(root/'episodes.pt')==receipt['sha256']
    data=torch.load(root/'episodes.pt',weights_only=False)
    assert sha256_file(root/'teachers.json')==data['teacher_manifest_sha256']
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    auth=json.loads((root/'responsibility.json').read_text())
    geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=a.device)
    geometry.assert_binding(data['geometry_binding'])
    parent_report=json.loads((root/a.parent/'report.json').read_text())
    parent,payload=load_element_retrieval_field(root/a.parent/'field.pt',expected_sha256=parent_report['field_sha256'],geometry=geometry,expected_query_encoder=json.loads((root/'native_query_encoder.json').read_text()),device=a.device)
    architecture=dict(payload['architecture'],structural_membership={'use_structure':not a.semantic_only,'seeds':4})
    field=ElementRetrievalField(**architecture).to(a.device)
    missing=field.load_state_dict(payload['state_dict'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('membership.') for k in missing.missing_keys)
    field.requires_grad_(False);field.membership.requires_grad_(True)
    field.membership.bind_geometry(geometry.xyz,field.observed)
    validate_episode_split(data['episodes'],data['proposal_views'],info['frames'],info['train_frames'],info['validation_frames'])
    indexed=[(i,e) for i,e in enumerate(data['episodes']) if e['split']=='train' and e['modality']!='text']
    indexed,_=source_region_query_tasks(indexed,data,[info['frames'].index(f) for f in info['train_frames']])
    images=[e for _,e in indexed if e['modality']=='image']
    gp=root/'grounded_text_supervision.pt';assert sha256_file(gp)==parent_report['grounded_text_supervision_sha256']
    grounded=torch.load(gp,weights_only=False)
    assert grounded['episode_sha256']==receipt['sha256'] and grounded['geometry_binding']==data['geometry_binding'] and grounded['train_frames']==info['train_frames']
    for key in ('positive_pixels','negative_pixels','positive_elements','negative_elements'):grounded[key]=grounded[key].to(a.device)
    tasks={'text':grounded['tasks'],'image':images}
    masks=data['masks'].flatten(1).to(a.device);rp=data['region_probability'].float().to(a.device)
    conflicts=training_query_conflicts(images,rp)
    mats={};support={};n=len(field.latent);pixels=auth['num_pixels']
    for v in sorted({e['target_view'] for group in tasks.values() for e in group}):
        assert info['frames'][v] in info['train_frames']
        r=auth['views'][v];path=root/r['relative_path'];assert sha256_file(path)==r['sha256']
        hit=torch.load(path,map_location=a.device,weights_only=False)
        pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();w=hit['base_weights'].float()
        mass=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w)
        mats[v]=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
        support[v]=torch.sparse.mm(mats[v],field.observed.float()[:,None])[:,0]
    queries={m:data['queries'][m].float().to(a.device) for m in tasks}
    with torch.no_grad():
        scores={m:field.native_cosine(q).detach() for m,q in queries.items()}
    def evidence(e,m):
        qi=e.get('text_index',e['query_region']);v=e['target_view']
        if m=='text':
            index=e['grounded_text_task'];pos=grounded['positive_pixels'][index];neg=grounded['negative_pixels'][index]
            rpos=grounded['positive_elements'][qi];rneg=grounded['negative_elements'][qi]
            return pos.float(),(pos^neg)&(support[v]>.5),rpos.float(),(rpos^rneg)&field.observed&data['seen'][v].to(a.device)
        target,known,rt,rk=episode_membership_targets(e,masks,rp,field.observed,support[v])
        ambiguous=conflicts.get(('image',qi))
        if ambiguous is not None:
            ambiguous=ambiguous&field.observed
            fraction=torch.sparse.mm(mats[v],ambiguous.float()[:,None])[:,0]/support[v].clamp_min(1e-8)
            known=known&(fraction<=.01);rk=rk&~ambiguous
        return target,known,rt,rk
    optimizer=torch.optim.Adam(field.membership.parameters(),lr=.001)
    trace=[];seed_audit=[];steps=a.epochs*max(map(len,tasks.values()))*2
    for step in range(steps):
        m=('text','image')[step%2];index=(step//2)%len(tasks[m]);e=tasks[m][index]
        qi=e.get('text_index',e['query_region']);v=e['target_view']
        if m=='image':assert info['frames'][int(data['proposal_views'][qi])] in info['train_frames']
        target,known,rt,rk=evidence(e,m)
        raw=scores[m][:,qi]
        probability=field.membership(raw,field.latent[:,field.native_semantic_dim:],field.observed,m).sigmoid()
        rendered=torch.sparse.mm(mats[v],(probability*field.observed)[:,None])[:,0]/support[v].clamp_min(1e-8)
        loss=membership_loss(probability,rt,rk)+membership_loss(rendered,target,known)
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step//2<len(tasks[m]):
            ids=field.membership.seed_indices(raw,field.observed)
            seed_audit.append({'modality':m,'query':qi,'view':v,'known_seed_count':int(rk[ids].sum()),'positive_seed_count':int((rk[ids]&(rt[ids]>.5)).sum()),'seed_count':len(ids)})
        if step%100==0 or step==steps-1:
            record={'step':step,'modality':m,'loss':float(loss.detach())};trace.append(record);print(json.dumps(record),flush=True)
    state={k:v.detach().cpu() for k,v in field.state_dict().items()}
    assert all(torch.equal(v,state[k]) for k,v in payload['state_dict'].items()),'parent state changed'
    with torch.no_grad():
        error=max(float((field.native_cosine(q)-parent.native_cosine(q)).abs().max()) for q in queries.values())
    assert error==0
    assert all(sha256_file(Path(k))==v for k,v in implementation.items()),'implementation changed during training'
    torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
    cold,_=load_element_retrieval_field(out/'field.pt',expected_sha256=sha256_file(out/'field.pt'),geometry=geometry,expected_query_encoder=payload['query_encoder'],device=a.device)
    with torch.no_grad():
        cold_error=max(float((cold.retrieve(q[:1],m)['scores']-field.retrieve(q[:1],m)['scores']).abs().max()) for m,q in queries.items())
    assert cold_error==0
    report={'schema':'structural_membership_control.v1','field_sha256':sha256_file(out/'field.pt'),'parent_field_sha256':parent_report['field_sha256'],'architecture':architecture,'source_only':True,'all_parent_state_exactly_preserved':True,'native_cosine_max_error':error,'cold_load_max_error':cold_error,'grounded_text_supervision_sha256':sha256_file(gp),'episode_sha256':receipt['sha256'],'steps':steps,'tasks':{m:len(x) for m,x in tasks.items()},'trace':trace,'automatic_seed_audit':seed_audit,'implementation':implementation,'benchmark_data_used':False,'limitations':['partial source targets','no scale hierarchy','semantic-only control shares geometry-based seed selection']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(out),'complete':True}),flush=True)
if __name__=='__main__':main()
