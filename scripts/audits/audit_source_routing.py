"""Separate source routing evidence from evaluation-query diagnostics."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_common import load_labels


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    output=Path(f'paper/artifacts/{a.scene}_source_routing_audit_20260916.json')
    if output.exists():raise FileExistsError(output)
    names=['grounded_semantic_transfer96','source_route_centered96','source_route_competition96','source_route_centered_competition96']
    parent=torch.load(root/names[0]/'field.pt',weights_only=False)
    identity=parent['query_encoder'];encoder=NativeThreeQueryEncoder(identity,a.device)
    _,categories,_,_=load_labels(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'),a.scene)
    queries=encoder.encode_text(categories).half().float()
    groups=json.loads((root/names[0]/'report.json').read_text())['groups'];phrases=[g['phrase'] for g in groups]
    probes=encoder.encode_text(['a picture of the '+s for s in phrases])
    r=json.loads((root/names[-1]/'report.json').read_text());vocabulary=r['vocabulary']
    source_queries=encoder.encode_text(vocabulary).half().float()
    models={}
    for name in names:
        report=json.loads((root/name/'report.json').read_text());assert sha256_file(root/name/'field.pt')==report['field_sha256']
        payload=torch.load(root/name/'field.pt',weights_only=False)
        assert payload['geometry_binding']==parent['geometry_binding']
        for k,v in parent['state_dict'].items():
            if k!='grounded_transfer.acceptance':assert torch.equal(v,payload['state_dict'][k]),k
        model=ElementRetrievalField(**payload['architecture']).to(a.device);model.load_state_dict(payload['state_dict']);model.requires_grad_(False)
        models[name]=model
    baseline=models[names[0]];rows=torch.arange(0,len(baseline.latent),137,device=a.device)
    with torch.no_grad():
        base_image=baseline.retrieve(queries,'image',rows=rows)['scores']
        base_cosine=baseline.native_cosine(queries)
        result={}
        for name,model in models.items():
            torch.testing.assert_close(model.native_cosine(queries),base_cosine,rtol=0,atol=0)
            torch.testing.assert_close(model.retrieve(queries,'image',rows=rows)['scores'],base_image,rtol=0,atol=0)
            head=model.grounded_transfer;index,accepted,score=head.route(queries)
            pi,pa,_=head.route(probes)
            si,sa,_=head.route(source_queries)
            unsupported=torch.tensor([s not in phrases for s in vocabulary],device=a.device)
            # A source phrase without an admitted head is not proven absent. This
            # measures forced assignment, not semantic false-positive ground truth.
            source=dict(template_probes=len(phrases),accepted_template_probes=int(pa.sum()),
                own_head_template_probes=int(((pi==torch.arange(len(phrases),device=a.device))&pa).sum()),
                unrepresented_phrases=int(unsupported.sum()),unrepresented_phrases_assigned=int((sa&unsupported).sum()))
            gates=[]
            for i,c in enumerate(categories):
                g=dict(query=c,source_phrase=phrases[int(index[i])],accepted=bool(accepted[i]),score=float(score[i]),threshold=float(head.acceptance[index[i]]))
                if head.routing is not None and len(head.routing_competitors):
                    q=F.normalize(F.normalize(queries[i:i+1],dim=-1)-head.routing_center,dim=-1)
                    comp=F.normalize(F.normalize(head.routing_competitors,dim=-1)-head.routing_center,dim=-1)
                    value,ci=(q@comp.T).max(1);g.update(competitor=r['competitors'][int(ci)],competitor_score=float(value))
                gates.append(g)
            result[name]=dict(field_sha256=sha256_file(root/name/'field.pt'),source_diagnostic=source,gates=gates)
    output.write_text(json.dumps(dict(scene=a.scene,source_and_evaluation_diagnostics_separate=True,runs=result,
        every_parent_tensor_except_acceptance_exact=True,cold_native_and_image_sample_exact=True,
        implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
    print(a.scene,{k:v['source_diagnostic'] for k,v in result.items()},flush=True)


if __name__=='__main__':main()
