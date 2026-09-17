"""Bounded source-region native reconstruction; source-view selection only.

This probes supervision-to-query access. Region-mean reconstruction is a proxy,
not equivalent to rendered element-cosine loss or physical instance supervision.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--root',type=Path);p.add_argument('--device',required=True);p.add_argument('--parent',default='exact_render_query_threshold96')
    a=p.parse_args();torch.set_num_threads(4)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/a.parent;ep=root/'caption_region_evidence.pt'
    receipt=json.loads((parent/'report.json').read_text());er=json.loads(ep.with_suffix('.json').read_text())
    assert sha256_file(parent/'field.pt')==receipt['field_sha256'] and sha256_file(ep)==er['sha256']
    payload=torch.load(parent/'field.pt',weights_only=False);e=torch.load(ep,weights_only=False)
    info=json.loads((root/'teachers.json').read_text())
    assert payload['geometry_binding']==e['geometry_binding'] and e['source_only'] and e['benchmark_data_used'] is False
    assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    assert all(info['frames'][int(v)] in e['train_frames'] for v in e['proposal_views'])
    arch=dict(payload['architecture'],native_semantic_residual={'relative_bound':.1})
    field=ElementRetrievalField(**arch).to(a.device)
    missing=field.load_state_dict(payload['state_dict'],strict=False)
    assert not missing.unexpected_keys and set(missing.missing_keys)=={'native_residual_decoder.weight','native_residual_decoder.bias'}
    field.requires_grad_(False)
    z=field.latent;observed=field.observed
    # Average the same supported elements for all variants; no element table added.
    means=[]
    for probability in e['region_probability'].split(32):
        weight=probability.float().to(a.device)*observed
        means.append(weight@z/weight.sum(-1,keepdim=True).clamp_min(1e-12))
    mean=torch.cat(means);target=F.normalize(e['native_descriptors'].float().to(a.device),dim=-1)
    base=field.native_decoder(mean)
    views=e['proposal_views'].to(a.device);unique=views.unique(sorted=True)
    # Held out from residual fitting only. The frozen parent has seen these views.
    val=(views==unique[-1]);train=~val
    assert train.sum()>2 and val.sum()>2
    center=mean[train].mean(0);scale=mean[train].std(0,unbiased=False).clamp_min(.01)
    x=torch.cat([(mean-center)/scale,torch.ones(len(mean),1,device=a.device)],-1)
    y=target-base
    trials=[];coeffs={}
    for ridge in (.1,1.,10.):
        regularizer=torch.eye(x.shape[1],device=a.device)*ridge;regularizer[-1,-1]=ridge*.01
        coefficients=torch.linalg.solve(x[train].T@x[train]/train.sum()+regularizer,x[train].T@y[train]/train.sum())
        coeffs[ridge]=coefficients
        delta=x@coefficients
        for bound in (0.,.1,.25,.5):
            gate=(bound*base.norm(dim=-1)/delta.norm(dim=-1).clamp_min(1e-12)).clamp_max(1)
            corrected=base+gate[:,None]*delta
            cos=(F.normalize(corrected,dim=-1)*target).sum(-1)
            trials.append(dict(ridge=ridge,relative_bound=bound,train_cosine=float(cos[train].mean()),validation_cosine=float(cos[val].mean())))
    selected=max(trials,key=lambda t:(t['validation_cosine'],-t['relative_bound']))
    for name,trial in [('native_residual_source_selected',selected),('native_residual_zero_control',dict(ridge=selected['ridge'],relative_bound=0.))]:
        out=root/name;out.mkdir(exist_ok=False)
        coefficients=coeffs[trial['ridge']]
        with torch.no_grad():
            field.native_residual_decoder.weight.copy_((coefficients[:-1]/scale[:,None]).T)
            field.native_residual_decoder.bias.copy_(coefficients[-1]-(center/scale)@coefficients[:-1])
        arch=dict(arch,native_semantic_residual={'relative_bound':trial['relative_bound']})
        field.native_semantic_residual=arch['native_semantic_residual']
        state={k:v.detach().cpu() for k,v in field.state_dict().items()}
        for key,value in payload['state_dict'].items():assert torch.equal(value,state[key]),key
        torch.save(dict(payload,architecture=arch,state_dict=state),out/'field.pt')
        report=dict(field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=receipt['field_sha256'],
            schema='bounded_native_semantic_residual.v1',source_only=True,benchmark_data_used=False,
            evidence_sha256=er['sha256'],all_parent_tensors_exact=True,query_encoder_unchanged=True,
            prompt_unchanged=True,image_unchanged=trial['relative_bound']==0,
            parent_report_sha256=sha256_file(parent/'report.json'),
            native_cosine_exactly_preserved=trial['relative_bound']==0,selection=trial,trials=trials,
            validation_frame=info['frames'][int(unique[-1])],train_regions=int(train.sum()),validation_regions=int(val.sum()),
            validation_contract='Residual-fit held-out source view; frozen field has seen this view. Mean feature cosine proxy, not rasterized IoU.',
            fitting_contract='Global native residual from source region-average latent; no target query or GT. No additional element state.',
            implementation_sources={f:sha256_file(f) for f in [__file__,'radio_gs/models/element_retrieval_field.py','radio_gs/models/bounded_native_residual.py']})
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(a.scene,selected,flush=True)


if __name__=='__main__':main()
