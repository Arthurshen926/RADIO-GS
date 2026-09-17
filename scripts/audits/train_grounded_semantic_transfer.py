"""Fit source-grounded extent heads; source-view validation selects the checkpoint."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    path=root/'caption_region_evidence.pt';receipt=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==receipt['sha256']
    evidence=torch.load(path,weights_only=False);info=json.loads((root/'teachers.json').read_text());assert evidence['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    lr=json.loads((root/'grounded_semantic_transfer96/report.json').read_text());linear=torch.load(root/'grounded_semantic_transfer96/field.pt',weights_only=False)
    assert sha256_file(root/'grounded_semantic_transfer96/field.pt')==lr['field_sha256'] and lr['evidence_sha256']==receipt['sha256']
    groups=lr['groups'];parent=root/'full_source_query_threshold_control';report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False);assert payload['geometry_binding']==evidence['geometry_binding']
    architecture=dict(payload['architecture'],grounded_transfer={'num_groups':len(groups),'hidden_dim':128})
    field=ElementRetrievalField(**architecture).to(a.device);missing=field.load_state_dict(payload['state_dict'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('grounded_transfer.') for k in missing.missing_keys)
    field.requires_grad_(False);head=field.grounded_transfer;head.decoder.requires_grad_(True)
    with torch.no_grad():
        head.prototypes.copy_(linear['state_dict']['grounded_transfer.prototypes'])
        head.acceptance.copy_(linear['state_dict']['grounded_transfer.acceptance'])
    z=head.features(field.latent).detach();prob=evidence['region_probability'].float().to(a.device);seen=evidence['seen'].to(a.device)
    heldout_frame=info['train_frames'][-1];heldout=info['frames'].index(heldout_frame)
    training={};validation={};disabled=[]
    for j,g in enumerate(groups):
        numerator=torch.zeros(len(z),device=a.device);denominator=torch.zeros_like(numerator)
        for vi in g['views']:
            ids=[i for i in g['regions'] if int(evidence['proposal_views'][i])==vi]
            target=prob[ids].amax(0)
            if vi==heldout:
                known=field.observed&seen[vi]
                if (target[known]>.5).any():validation[j]=(known,target)
            else:numerator+=target;denominator+=seen[vi]
        target=numerator/denominator.clamp_min(1);known=field.observed&(denominator>0)
        positive=torch.where(known&(target>.7))[0];negative=torch.where(known&(target<.1))[0]
        if min(len(positive),len(negative))<8:
            disabled.append(j);head.acceptance[j]=2.;continue
        training[j]=(positive,negative,target)
    validation={j:v for j,v in validation.items() if j in training}
    assert training and validation
    optimizer=torch.optim.Adam(head.decoder.parameters(),lr=.001);ids=list(training);history=[];best=-1.;best_state=None;best_step=None
    def validate():
        values=[]
        with torch.no_grad():
            prediction=torch.cat([head.decoder(x) for x in z.split(16384)])>0
            for j,(known,target) in validation.items():
                truth=target>.5;pred=prediction[:,j]
                values.append(float((pred&truth&known).sum()/((pred|truth)&known).sum().clamp_min(1)))
        return sum(values)/len(values)
    for step in range(2000):
        j=ids[step%len(ids)];positive,negative,target=training[j]
        rows=torch.cat([positive[torch.randint(len(positive),(256,),device=a.device)],negative[torch.randint(len(negative),(256,),device=a.device)]])
        logits=head.decoder(z[rows])[:,j];loss=F.binary_cross_entropy_with_logits(logits,target[rows])
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%400==0:
            score=validate();history.append(dict(step=step+1,source_validation_iou=score,last_balanced_bce=float(loss.detach())))
            if score>best:best=score;best_state={k:v.detach().cpu().clone() for k,v in head.decoder.state_dict().items()};best_step=step+1
            print(a.scene,history[-1],flush=True)
    head.decoder.load_state_dict(best_state)
    state={k:v.detach().cpu() for k,v in field.state_dict().items()}
    for key,value in payload['state_dict'].items():assert torch.equal(value,state[key]),key
    out=root/'grounded_semantic_transfer_mlp96';out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
    result=dict(lr,schema='grounded_semantic_transfer_mlp.v1',field_sha256=sha256_file(out/'field.pt'),
        parent_field_sha256=report['field_sha256'],linear_gate_parent_sha256=lr['field_sha256'],source_only=True,benchmark_data_used=False,
        hidden_dim=128,steps=2000,selected_step=best_step,source_validation_iou=best,validation_teacher_frame=heldout_frame,
        validation_scope='source teacher-label view held out from head fitting; base field already saw this source RGB; not an unseen-scene test',
        disabled_groups=disabled,training_groups=ids,validation_groups=list(validation),history=history,
        source_fit=None,ridge=None,all_parent_tensors_unchanged=True,image_prompt_unchanged=True,
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(out,flush=True)


if __name__=='__main__':main()
