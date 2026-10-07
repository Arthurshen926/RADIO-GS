"""Matched inferred-support control: one optional mask, unchanged member model."""
import argparse
import json
from pathlib import Path

import torch

from .contracts import QueryPacket,Support
from .public_lerf import score
from .storage import load_field,save_field,sha256


@torch.no_grad()
def package(config,job,output,device='cuda'):
    field,manifest=load_field(job['parent_field'],device)
    audit=Path(job['support_audit']);record=json.loads((audit/'report.json').read_text())
    if record['field_sha256']!=manifest['files']['field.pt']['sha256']:
        raise ValueError('support audit field differs')
    if record['exposure_sha256']!=sha256(audit/'training_exposure.pt'):
        raise ValueError('support exposure changed')
    exposure=torch.load(audit/'training_exposure.pt',map_location='cpu',weights_only=True)
    eligible=exposure['eligible'].to(device)
    if eligible.shape!=field.observed.shape or eligible.dtype!=torch.bool or not eligible.any():
        raise ValueError('invalid anchor domain')
    if bool((eligible&~field.observed).any()):raise ValueError('support is outside observed carrier')
    # Representative registration replay spans all observed elements, including
    # rows outside the inferred-anchor domain. Full development scores follow.
    observed=torch.where(field.observed)[0]
    ids=observed[torch.linspace(0,len(observed)-1,16,device=device).long()]
    packets=[QueryPacket('point_3d',positive=Support(i.reshape(1),torch.ones(1,device=device)),exact_positive=True) for i in ids]
    packets.append(QueryPacket('point_2d',positive=Support(ids,torch.ones(len(ids),device=device))))
    before=[field.query(packet).membership_scores for packet in packets]
    field.anchor_eligible=eligible
    for packet,reference in zip(packets,before):
        if not torch.equal(field.query(packet).membership_scores,reference):raise ValueError('registered support changed')
    exported=save_field(field,job['field'],dict(parent_field_sha256=manifest['files']['field.pt']['sha256'],
        support_audit_sha256=sha256(audit/'report.json'),observations_sha256=record['observations_sha256'],
        operation='Restrict inferred anchors to train-only reliable interior support; complete observed member carrier retained.',
        original_tensors_unchanged=True,registered_replay_exact=True,extra_3d_labels=False,benchmark_queries_used=False))
    report=dict(job=job,config=config,export=exported,registered_replay_cases=len(packets),registered_replay_exact=True)
    root=Path(job['field']).parents[1]
    (root/'support_package.json').write_text(json.dumps(report,indent=2)+'\n')
    return dict(eligible_anchors=exported['eligible_anchors'],bytes=exported['files']['field.pt']['bytes'])


@torch.no_grad()
def predict(config,job,output,device='cuda'):
    root=Path(output)/job['name'];root.mkdir(parents=True,exist_ok=False)
    previous=Path(job['reference_prediction']);receipt=json.loads((previous/'prediction.json').read_text())
    if sha256(previous/'predictions.pt')!=receipt['predictions_sha256']:raise ValueError('reference prediction changed')
    data=torch.load(previous/'predictions.pt',map_location='cpu',weights_only=True)
    field,manifest=load_field(job['field'],device)
    if manifest['provenance'].get('parent_field_sha256')!=receipt['field_sha256']:
        raise ValueError('reference prediction is not the unchanged parent field')
    for key,value in data['geometry'].items():
        if not torch.equal(getattr(field,key).cpu(),value):raise ValueError('geometry differs')
    if not torch.equal(field.observed.cpu(),data['observed']):raise ValueError('member carrier differs')
    background=field.semantic.score(data['generic_negative_vectors']).max(1).values
    scores=[];selected=[];diagnostics=[]
    for column,(text,vector) in enumerate(zip(data['texts'],data['native_query_vectors'])):
        result=field.query(QueryPacket('text',task='union',semantic=vector.to(device)))
        native=torch.sigmoid(10*(result.semantic_scores-background)).cpu()
        if not torch.equal(native,data['scores']['native_generic_relevance'][:,column]):
            raise ValueError('native response changed')
        scores.append(result.membership_scores.cpu());selected.append(result.selected.cpu())
        diagnostics.append(dict(text=text,selected=int(result.selected.sum()),**result.diagnostics))
        print('support control',job['name'],text,flush=True)
    payload=dict(texts=data['texts'],geometry=data['geometry'],observed=data['observed'],
        scores=dict(native_generic_relevance=data['scores']['native_generic_relevance'],shared_membership=torch.stack(scores,1)),
        selected=torch.stack(selected,1))
    torch.save(payload,root/'predictions.pt')
    report=dict(schema='radio_gs.query_field_support_prediction.v1',job=job,configuration=config,
        predictions_sha256=sha256(root/'predictions.pt'),field_sha256=manifest['files']['field.pt']['sha256'],
        reference_prediction_sha256=receipt['predictions_sha256'],native_response_bit_exact=True,
        member_carrier_unchanged=True,runtime_source_access=False,legacy_calibration_reused=False,diagnostics=diagnostics)
    (root/'prediction.json').write_text(json.dumps(report,indent=2)+'\n')
    return {key:value for key,value in report.items() if key!='diagnostics'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['package','predict','score'])
    parser.add_argument('--config',required=True);parser.add_argument('--job',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args();torch.set_num_threads(4)
    config=json.loads(Path(args.config).read_text());job=next(j for j in config['jobs'] if j['name']==args.job)
    print(json.dumps({'package':package,'predict':predict,'score':score}[args.stage](config,job,args.output,args.device),indent=2))
