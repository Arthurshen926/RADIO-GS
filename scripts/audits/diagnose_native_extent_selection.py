"""Trace query-only extent selection; benchmark names are diagnostic inputs only."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.v4.evaluation.lerf_common import load_labels
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    out=Path(f'paper/artifacts/{a.scene}_native_extent_selection_trace_20260916.json')
    if out.exists():raise FileExistsError(out)
    path=root/'native_identity_global96/field.pt';payload=torch.load(path,weights_only=False)
    field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    encoder=NativeThreeQueryEncoder(json.loads((root/'native_query_encoder.json').read_text()),a.device)
    _,categories,_,_=load_labels(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'),a.scene)
    groups=json.loads((root/'grounded_semantic_transfer96/report.json').read_text())['groups']
    queries=encoder.encode_text(categories).half().float();head=field.grounded_transfer
    with torch.no_grad():
        threshold=field.native_query_thresholds['text'](queries)
        positive=(field.native_cosine(queries)>threshold)&field.observed[:,None]
        intersection=torch.zeros(len(categories),head.num_proposals,device=a.device)
        mass=torch.zeros(head.num_proposals,device=a.device)
        for rows in torch.arange(len(field.latent),device=a.device).split(4096):
            f=head.features(field.latent[rows]);mask=((head.decoder(f)+head.residual(f))>.5)&field.observed[rows,None]
            intersection+=positive[rows].float().T@mask.float();mass+=mask.sum(0)
        support=positive.sum(0);overlap=intersection/(support[:,None]+mass-intersection).clamp_min(1)
        mean=F.normalize(queries,dim=-1)@head.native_extent_means.T
        crop=F.normalize(queries,dim=-1)@F.normalize(head.proposal_embeddings,dim=-1).T
        group,accepted,_=head.route(queries);eligible=head.proposal_groups[:,group].T
        records=[]
        for i,c in enumerate(categories):
            policies={}
            for name,scores in [('mean',mean),('overlap',overlap),('crop',crop)]:
                for restricted in [True,False]:
                    values=scores[i].masked_fill(~eligible[i],-torch.inf) if restricted else scores[i]
                    k=int(values.argmax());accept=bool(accepted[i]) if restricted else True
                    if name=='mean' and not restricted:accept &= bool(values[k]>threshold[i])
                    if name=='overlap':accept &= bool(values[k]>0)
                    policies[name+('_restricted' if restricted else '_global')]=dict(candidate=k,accepted=accept,score=float(values[k]),
                        candidate_elements=int(mass[k]),native_support_precision=float(intersection[i,k]/mass[k].clamp_min(1)),
                        native_support_recall=float(intersection[i,k]/support[i].clamp_min(1)),
                        source_concepts=[groups[j]['phrase'] for j in head.proposal_groups[k].nonzero().flatten().tolist()])
            records.append(dict(category=c,source_gate=bool(accepted[i]),source_concept=groups[int(group[i])]['phrase'],
                native_threshold=float(threshold[i]),native_positive_elements=int(support[i]),policies=policies))
    out.write_text(json.dumps(dict(schema='native_extent_selection_trace.v1',diagnostic_only=True,benchmark_queries_used=True,
        benchmark_masks_used=False,field_sha256=sha256_file(path),implementation_sha256=sha256_file(__file__),records=records),indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
