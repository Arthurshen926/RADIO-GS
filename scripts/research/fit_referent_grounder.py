"""Matched referent learners, selected only on independent calibration rooms."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.querying.scene_referent_grounding import ReferentGrounder,region_geometry
from radio_gs.querying.referent_identity import native_logit_scale
from scripts.research.prepare_referent_grounding import ENCODER,ROOT


def pack_same_class_queries(ids,rows):
    """Retain every scheduled example, interleaving different instances per class."""
    groups={}
    for index in ids:
        row=rows[index]
        groups.setdefault(row['category'],{}).setdefault(row['raw_object_id'],[]).append(index)
    packed=[]
    for instances in groups.values():
        while any(instances.values()):
            for bucket in instances.values():
                if bucket:packed.append(bucket.pop())
    return np.asarray(packed,dtype=np.int64)


def read_scene(scene,device,root=ROOT):
    folder=root/'maps'/scene
    descriptors=np.load(folder/'descriptors.npy').astype(np.float32)
    descriptors/=np.maximum(np.linalg.norm(descriptors,axis=1,keepdims=True),1e-8)
    membership=sparse.load_npz(folder/'membership.npz');visibility=sparse.load_npz(folder/'visibility.npz')
    frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(folder/'region_offsets.npy')))
    geometry,available=region_geometry(membership,visibility,frames,np.load(folder/'mesh_xyz.npy'))
    ids=np.flatnonzero(available)
    language=np.load(root/'language'/(scene+'.npz'))
    pooled=language['pooled'].astype(np.float32)
    pooled/=np.linalg.norm(pooled,axis=1,keepdims=True)
    rows=json.loads((root/'language'/(scene+'.json')).read_text())
    targets=np.load(folder/'targets.npz');columns={int(x):i for i,x in enumerate(targets['instance_ids'])}
    reward=np.stack([targets['scope_iou'][ids,columns[q['raw_object_id']+1]] for q in rows])
    area=np.asarray(membership.sum(0)).ravel()[ids]
    purity=np.stack([targets['region_intersection'][ids,columns[q['raw_object_id']+1]]/area for q in rows])
    tensors={k:torch.from_numpy(v).to(device) for k,v in dict(visual=descriptors[ids],geometry=geometry[ids],
        tokens=language['tokens'].astype(np.float32),valid=language['valid'],pooled=pooled,reward=reward,purity=purity).items()}
    tensors.update(scene=scene,rows=rows,ids=ids)
    return tensors


@torch.inference_mode()
def evaluate(model,scenes,batch_size=8):
    model.eval();records=[]
    for data in scenes:
        guesses=[];native=[];identity=[];native_identity=[];seeds=[];native_seeds=[]
        for start in range(0,len(data['rows']),batch_size):
            sl=slice(start,start+batch_size)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scores=model(data['visual'],data['geometry'],data['tokens'][sl],data['valid'][sl],data['pooled'][sl])
            best=scores.argmax(1);baseline=(data['pooled'][sl]@data['visual'].T).argmax(1)
            reward=data['reward'][sl];row=torch.arange(len(best),device=best.device)
            guesses.extend(reward[row,best].cpu().tolist());native.extend(reward[row,baseline].cpu().tolist())
            identity.extend(data['purity'][sl][row,best].cpu().tolist())
            native_identity.extend(data['purity'][sl][row,baseline].cpu().tolist())
            seeds.extend(data['ids'][best.cpu().numpy()].tolist());native_seeds.extend(data['ids'][baseline.cpu().numpy()].tolist())
        records.append(dict(scene=data['scene'],queries=len(guesses),iou=float(np.mean(guesses)),
            native_iou=float(np.mean(native)),finite_family_oracle=float(data['reward'].max(1).values.mean()),
            no_candidate_target_support=int((data['reward'].max(1).values==0).sum()),
            majority_identity_accuracy=float(np.mean(np.array(identity)>.5)),
            native_majority_identity_accuracy=float(np.mean(np.array(native_identity)>.5)),
            per_query_iou=guesses,per_query_native_iou=native,per_query_purity=identity,
            per_query_native_purity=native_identity,per_query_seed=seeds,per_query_native_seed=native_seeds))
    return dict(scene_macro_iou=float(np.mean([r['iou'] for r in records])),
        scene_macro_native_iou=float(np.mean([r['native_iou'] for r in records])),
        scene_macro_identity_accuracy=float(np.mean([r['majority_identity_accuracy'] for r in records])),
        scene_macro_native_identity_accuracy=float(np.mean([r['native_majority_identity_accuracy'] for r in records])),scenes=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--condition',choices=['local','scene'],required=True)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--data-root',type=Path,default=ROOT);p.add_argument('--output-name')
    p.add_argument('--target',choices=['scope','identity'],default='scope')
    p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--objective',choices=['forward','bidirectional','bidirectional_posterior'],default='forward')
    p.add_argument('--eligibility',choices=['listener','named'],default='listener')
    p.add_argument('--include-native-checkpoint',action='store_true')
    p.add_argument('--batch-organization',choices=['random','category'],default='random')
    a=p.parse_args();torch.set_num_threads(3);root=a.data_root
    torch.manual_seed(20261005);np.random.seed(20261005);device='cuda'
    design=json.loads((root/'design.json').read_text())
    groups={key:[read_scene(s,device,root) for s in design[key] if (root/'language'/(s+'.npz')).exists()]
            for key in ['fit','calibration','validation']}
    architecture=dict(context=a.condition=='scene',width=128,text_width=1152,native_width=1536,native_scale=native_logit_scale(ENCODER))
    model=ReferentGrounder(**architecture).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    out=root/(a.output_name or a.condition);out.mkdir(exist_ok=True);start=time.monotonic();history=[];best=-1
    rng=np.random.default_rng(20261005)
    fit_rows=sum(sum(q['listener_correct'] for q in d['rows']) for d in groups['fit'])
    eligible_rows=[];logical_counts=[]
    for data in groups['fit']:
        listener=[j for j,q in enumerate(data['rows']) if q['listener_correct'] and float(data['reward'][j].max())>0]
        eligible_rows.append(np.array([j for j in listener if a.eligibility=='listener' or data['rows'][j]['mentions_target_class']]))
        logical_counts.append(len(listener))
    target_description=('decoded-set IoU^4' if a.target=='scope' else 'requested instance fraction of seed region^4; completeness not in target')
    if a.objective!='forward':target_description+='; query-to-region and region-to-query weighted cross entropy'
    if a.objective=='bidirectional_posterior':target_description+='; reverse supervision uses query-normalized regional log probabilities'
    def save_checkpoint(epoch):
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},architecture=architecture,epoch=epoch,
            training_condition=design.get('training_condition','additional NR3D language and official3D instance supervision, disjoint rendered training rooms'),
            fit=design['fit'],calibration=design['calibration'],target=target_description,
            objective=a.objective,eligibility=a.eligibility,batch_organization=a.batch_organization),out/'model.pt')
    if a.include_native_checkpoint:
        calibration=evaluate(model,groups['calibration'],a.microbatch);best=calibration['scene_macro_iou']
        history.append(dict(epoch=0,loss=None,calibration=calibration));save_checkpoint(0)
    for epoch in range(1,a.epochs+1):
        tasks=[]
        for i,data in enumerate(groups['fit']):
            ids=eligible_rows[i].copy()
            rng.shuffle(ids)
            # Keep the original room/query budget; excluded descriptions are
            # replaced only by already eligible training descriptions.
            if a.eligibility=='named':ids=np.resize(ids,logical_counts[i])
            if a.batch_organization=='category':ids=pack_same_class_queries(ids,data['rows'])
            tasks.extend((i,ids[j:j+8]) for j in range(0,len(ids),8))
        rng.shuffle(tasks);model.train();losses=[]
        for index,ids in tasks:
            data=groups['fit'][index];ids=torch.as_tensor(ids,device=device)
            target=data['reward' if a.target=='scope' else 'purity'][ids].pow(4);target/=target.sum(1,keepdim=True)
            optimizer.zero_grad(set_to_none=True)
            loss_value=0.;joint_scores=[]
            for offset in range(0,len(ids),a.microbatch):
                selected=ids[offset:offset+a.microbatch]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    scores=model(data['visual'],data['geometry'],data['tokens'][selected],data['valid'][selected],data['pooled'][selected])
                if a.objective!='forward':joint_scores.append(scores.float())
                else:
                    loss=-(target[offset:offset+a.microbatch]*F.log_softmax(scores.float(),dim=-1)).sum()/len(ids)
                    loss.backward();loss_value+=float(loss)
            if a.objective!='forward':
                scores=torch.cat(joint_scores)
                regional_log_probability=F.log_softmax(scores,dim=1)
                forward=-(target*regional_log_probability).sum()/len(ids)
                # The same mass weights both directions. Near-zero-quality
                # candidates do not become strong labels after column scaling;
                # multiple descriptions of one instance stay multiple positives.
                reverse_scores=regional_log_probability if a.objective=='bidirectional_posterior' else scores
                reverse=-(target*F.log_softmax(reverse_scores,dim=0)).sum()/len(ids)
                loss=(forward+reverse)*.5;loss.backward();loss_value=float(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss_value)
        calibration=evaluate(model,groups['calibration'],a.microbatch);value=calibration['scene_macro_iou']
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),calibration=calibration))
        if value>best:
            best=value
            save_checkpoint(epoch)
        print(a.condition,'epoch',epoch,'loss',round(np.mean(losses),4),'calibration',round(value,5),'best',round(best,5),flush=True)
    saved=torch.load(out/'model.pt',map_location=device,weights_only=False);model.load_state_dict(saved['state_dict'])
    report=dict(condition=a.condition,supervision_target=a.target,data_root=str(root),microbatch=a.microbatch,
        objective=a.objective,eligibility=a.eligibility,training_eligible_unique_records=sum(map(len,eligible_rows)),
        batch_organization=a.batch_organization,
        training_query_presentations_per_epoch=sum(logical_counts),include_native_checkpoint=a.include_native_checkpoint,
        architecture=architecture,epochs=a.epochs,selected_epoch=saved['epoch'],
        training_listener_correct_records=fit_rows,
        excluded_training_zero_scope_support=sum(sum(q['listener_correct'] and float(d['reward'][i].max())==0 for i,q in enumerate(d['rows'])) for d in groups['fit']),
        all_calibration_and_validation_queries_retained=True,calibration=evaluate(model,groups['calibration'],a.microbatch),
        validation=evaluate(model,groups['validation'],a.microbatch),history=history,seconds=time.monotonic()-start,
        parameter_count=sum(p.numel() for p in model.parameters()),benchmark_labels_used=False,
        native_coordinates_changed=False,new_instance_scope_family=False)
    (out/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'training_fit_diagnosis.json').write_text(json.dumps(evaluate(model,groups['fit'],a.microbatch),indent=2)+'\n')
    print('validation',report['validation']['scene_macro_iou'],'native',report['validation']['scene_macro_native_iou'],flush=True)


if __name__=='__main__':main()
