"""Bound source-measured native text encoding drift at the concept gate.

No target query/GT is read. This only repairs numerical source-template admission;
it cannot certify stability for arbitrary future queries on a decision boundary.
"""
import argparse,json,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.functional_query_encoder import FunctionalThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--parent',default='exact_render_query_threshold96');p.add_argument('--name',default='source_precision_gate96');a=p.parse_args();torch.set_num_threads(4)
    root=a.root;path=root/a.parent/'field.pt';receipt=json.loads(path.with_name('report.json').read_text());assert sha256_file(path)==receipt['field_sha256']
    payload=torch.load(path,map_location='cpu',weights_only=False);field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    head=field.grounded_transfer;assert head is not None
    groups_path=root/'source_phrase_groups.json';groups=json.loads(groups_path.read_text())['groups'];assert len(groups)==len(head.prototypes)
    ep_path=root/'episodes.pt';ep_receipt=json.loads(ep_path.with_suffix('.json').read_text());assert sha256_file(ep_path)==ep_receipt['sha256'];ep=torch.load(ep_path,map_location='cpu',weights_only=False)
    assert ep['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and ep['geometry_binding']==payload['geometry_binding']
    cached=ep['queries']['text'].float().to(a.device);cached_phrases=ep['queries']['queries'];assert len(cached)==len(cached_phrases)
    phrases=list(dict.fromkeys([g['phrase'] for g in groups]+['a photo of a '+g['phrase'] for g in groups]+cached_phrases))
    encoder=FunctionalThreeQueryEncoder(payload['query_encoder'],a.device)
    with torch.no_grad():
        batch=encoder.encode_text(phrases);single=torch.cat([encoder.encode_text([q]) for q in phrases]);variants=[batch,batch.half().float(),single,single.half().float()]
        center=head.routing_center if head.routing is not None else torch.zeros_like(head.prototypes[0])
        prototype=F.normalize(F.normalize(head.prototypes,dim=-1)-center,dim=-1)
        def similarity(x):return F.normalize(F.normalize(x,dim=-1)-center,dim=-1)@prototype.T
        values=torch.stack([similarity(q) for q in variants]);low=values.amin(0);high=values.amax(0)
        for i,phrase in enumerate(cached_phrases):
            index=phrases.index(phrase);value=similarity(cached[i:i+1])[0];low[index]=torch.minimum(low[index],value);high[index]=torch.maximum(high[index],value)
        # The original threshold was fitted in a different batch: exactly one
        # photo template per group. Large mixed batches and singleton calls do
        # not bound that FP16 execution shape. Reproduce its actual encodings;
        # never invent a larger tolerance or remove the admission assertion.
        construction_templates=['a photo of a '+g['phrase'] for g in groups]
        construction_values=similarity(encoder.encode_text(construction_templates))
        for i,phrase in enumerate(construction_templates):
            index=phrases.index(phrase)
            low[index]=torch.minimum(low[index],construction_values[i])
            high[index]=torch.maximum(high[index],construction_values[i])
        # Per-prototype largest observed drift, plus FP32 dot-product roundoff.
        margin=(high-low).amax(0)+8*torch.finfo(torch.float32).eps
        original=head.acceptance.clone();head.acceptance.sub_(margin)
        checks=[]
        for i,g in enumerate(groups):
            index=phrases.index('a photo of a '+g['phrase'])
            checks.append(dict(phrase=g['phrase'],old_threshold=float(original[i]),new_threshold=float(head.acceptance[i]),measured_margin=float(margin[i]),minimum_template_cosine=float(low[index,i]),template_numerically_admitted=bool(low[index,i]>=head.acceptance[i])))
        if not all(x['template_numerically_admitted'] for x in checks):raise ValueError('source template admission still inconsistent; inspect rather than expanding tolerance')
        state={k:v.detach().cpu() for k,v in field.state_dict().items()}
        for key,value in payload['state_dict'].items():
            if key!='grounded_transfer.acceptance':assert torch.equal(value,state[key]),key
    out=root/a.name;out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source.py')
    torch.save(dict(payload,state_dict=state),out/'field.pt')
    result=dict(schema='source_native_gate_precision.v2',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=receipt['field_sha256'],episode_sha256=ep_receipt['sha256'],groups_sha256=sha256_file(groups_path),implementation_sha256=sha256_file(__file__),source_only=True,benchmark_data_used=False,all_other_state_exact=True,query_encoder_unchanged=True,geometry_unchanged=True,phrases=phrases,checks=checks,maximum_margin=float(margin.max()),construction_template_batch_size=len(construction_templates),policy='per-prototype maximum source encoding cosine range across mixed batch/singleton FP32/FP16, cached training queries and original construction-template batch, plus eight FP32 eps',future_query_stability_guaranteed=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(out,result['maximum_margin'],flush=True)

if __name__=='__main__':main()
