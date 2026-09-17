"""Source-grounded extent heads on frozen fields, with source-only query gating."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.functional_query_encoder import FunctionalThreeQueryEncoder as NativeThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.scope_union_lifting import lift_mask_union


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    p.add_argument('--evidence',default='caption_region_evidence.pt');p.add_argument('--group-report',default='source_membership_balanced32/report.json');p.add_argument('--name',default='grounded_semantic_transfer96')
    p.add_argument('--root',type=Path);a=p.parse_args();torch.set_num_threads(4)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    path=root/a.evidence;receipt=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==receipt['sha256']
    evidence=torch.load(path,weights_only=False)
    group_report=json.loads((root/a.group_report).read_text())
    assert group_report['evidence_sha256']==receipt['sha256']
    groups=group_report['groups'];phrases=[g['phrase'] for g in groups]
    parent=root/'full_source_query_threshold_control';report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False);assert payload['geometry_binding']==evidence['geometry_binding']
    architecture=dict(payload['architecture'],grounded_transfer={'num_groups':len(groups)})
    field=ElementRetrievalField(**architecture).to(a.device);missing=field.load_state_dict(payload['state_dict'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('grounded_transfer.') for k in missing.missing_keys)
    field.requires_grad_(False);head=field.grounded_transfer
    probability=evidence['region_probability'].float().to(a.device);seen=evidence['seen'].to(a.device)
    z=head.features(field.latent);x=torch.cat([z,torch.ones(len(z),1,device=a.device)],1)
    metrics=[];ridge=.001
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    mappings={}
    def union_target(members,vi):
        if len(members)==1:return probability[members[0]]
        if vi not in mappings:
            rec=auth['views'][vi];hp=root/rec['relative_path'];assert sha256_file(hp)==rec['sha256']
            hit=torch.load(hp,map_location=a.device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();w=hit['base_weights'].float();pixels=evidence['masks'][0].numel()
            mass=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w)
            mappings[vi]=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(pixels,len(z))).coalesce()
        result,visible=lift_mask_union(mappings[vi],evidence['masks'][members].to(a.device))
        assert torch.equal(visible,seen[vi])
        return result
    with torch.no_grad():
        for j,g in enumerate(groups):
            numerator=torch.zeros(len(z),device=a.device);denominator=torch.zeros_like(numerator)
            for vi in g['views']:
                members=[i for i in g['regions'] if int(evidence['proposal_views'][i])==vi]
                numerator+=union_target(members,vi);denominator+=seen[vi]
            target=numerator/denominator.clamp_min(1);known=(denominator>0)&field.observed
            positive=known&(target>.5);negative=known&~positive
            assert positive.any() and negative.any()
            weight=positive.float()/(2*positive.sum())+negative.float()/(2*negative.sum())
            weighted=x*weight[:,None]
            lhs=x.T@weighted;regularizer=torch.eye(x.shape[1],device=a.device)*ridge;regularizer[-1,-1]=0
            coefficients=torch.linalg.solve(lhs+regularizer,weighted.T@target)
            head.decoder.weight[j].copy_(coefficients[:-1]);head.decoder.bias[j].copy_(coefficients[-1])
            predicted=x@coefficients>.5
            iou=float((predicted&positive).sum()/((predicted&known)|positive).sum().clamp_min(1))
            metrics.append(dict(phrase=g['phrase'],source_known_iou=iou,positive_elements=int(positive.sum()),known_elements=int(known.sum())))
        head.prototypes.copy_(evidence['text_embeddings'][[g['source_query_index'] for g in groups]].float().to(a.device))
    encoder=NativeThreeQueryEncoder(payload['query_encoder'],a.device)
    template=encoder.encode_text(['a photo of a '+p for p in phrases])
    with torch.no_grad():
        # At least as close as a source phrase with the training photo template.
        # This gate is fixed from source text pairs; no benchmark phrase/GT tuning.
        acceptance=(F.normalize(template,dim=-1)*F.normalize(head.prototypes,dim=-1)).sum(-1)
        head.acceptance.copy_(acceptance)
    state={k:v.detach().cpu() for k,v in field.state_dict().items()}
    for key,value in payload['state_dict'].items():assert torch.equal(value,state[key]),key
    out=root/a.name;out.mkdir(exist_ok=False)
    torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
    r=dict(report,schema='grounded_semantic_transfer.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=report['field_sha256'],
        group_lifting='pixel_union_before_projection',union_implementation_sha256=sha256_file('radio_gs/training/scope_union_lifting.py'),evidence_sha256=receipt['sha256'],groups=groups,source_fit=metrics,source_gate=acceptance.cpu().tolist(),ridge=ridge,
        gate_policy='query/prototype cosine >= source bare/photo-template cosine; otherwise original text readout',
        source_only=True,benchmark_data_used=False,all_parent_tensors_unchanged=True,native_cosine_exactly_preserved=True,
        native_text_image_max_error=None,image_prompt_unchanged=True,
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(a.scene,sum(m['source_known_iou'] for m in metrics)/len(metrics),acceptance.min().item(),acceptance.max().item(),flush=True)


if __name__=='__main__':main()
