"""Source-only semantic routing controls; every extent and field tensor frozen."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'grounded_semantic_transfer96'
    report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False)
    paths=[root/'caption_region_evidence.pt',root/'fixed_vocab_region_evidence.pt']
    evidence=[]
    for path in paths:
        receipt=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==receipt['sha256']
        e=torch.load(path,weights_only=False);assert e['source_only'] and not e['benchmark_data_used']
        assert e['geometry_binding']==payload['geometry_binding'];evidence.append(e)
    vocabulary={}
    for e in evidence:
        for phrase,embedding in zip(e['phrases'],e['text_embeddings']):vocabulary.setdefault(phrase,embedding.float())
    phrases=[g['phrase'] for g in report['groups']]
    alternatives=sorted(set(vocabulary)-set(phrases))
    prototypes=payload['state_dict']['grounded_transfer.prototypes'].to(a.device)
    center=F.normalize(torch.stack(list(vocabulary.values())).to(a.device),dim=-1).mean(0)
    encoder=NativeThreeQueryEncoder(payload['query_encoder'],a.device)
    templates=encoder.encode_text(['a photo of a '+s for s in phrases])
    # Held-out template wording diagnostic: source concept identity, not target labels.
    probes=encoder.encode_text(['a picture of the '+s for s in phrases])
    baseline=ElementRetrievalField(**payload['architecture']).to(a.device)
    baseline.load_state_dict(payload['state_dict']);baseline.requires_grad_(False)
    controls={}
    for name,centered,competitors in [('centered',True,False),('competition',False,True),('centered_competition',True,True)]:
        config=dict(num_competitors=len(alternatives) if competitors else 0)
        architecture=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],routing=config))
        model=ElementRetrievalField(**architecture).to(a.device);missing=model.load_state_dict(payload['state_dict'],strict=False)
        assert set(missing.missing_keys)=={'grounded_transfer.routing_center','grounded_transfer.routing_competitors'} and not missing.unexpected_keys
        model.requires_grad_(False);head=model.grounded_transfer
        with torch.no_grad():
            if centered:head.routing_center.copy_(center)
            if competitors:head.routing_competitors.copy_(torch.stack([vocabulary[s] for s in alternatives]).to(a.device))
            head.acceptance.copy_((F.normalize(templates-head.routing_center,dim=-1)*F.normalize(F.normalize(prototypes,dim=-1)-head.routing_center,dim=-1)).sum(-1))
            indices,accepted,_=head.route(probes)
            source=dict(probes=len(phrases),accepted=int(accepted.sum()),correct_accepted=int(((indices==torch.arange(len(phrases),device=a.device))&accepted).sum()))
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        for key,value in payload['state_dict'].items():
            if key!='grounded_transfer.acceptance':assert torch.equal(value,state[key]),key
        out=root/('source_route_'+name+'96');out.mkdir(exist_ok=False)
        torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
        r=dict(report,schema='source_routing_control.v1',parent_field_sha256=report['field_sha256'],field_sha256=sha256_file(out/'field.pt'),
               routing_policy=name,source_only=True,benchmark_data_used=False,all_extent_and_parent_field_tensors_unchanged=True,
               all_parent_tensors_unchanged=False,source_gate=head.acceptance.cpu().tolist(),source_template_probe=source,
               vocabulary=sorted(vocabulary),competitors=alternatives if competitors else [],evidence_files={str(x):sha256_file(x) for x in paths},
               implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'),
               gate_policy='source-centered cosine and photo-template acceptance; optional non-admitted source phrases compete, force old-readout fallback')
        (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');controls[name]=source
    print(a.scene,controls,flush=True)


if __name__=='__main__':main()
