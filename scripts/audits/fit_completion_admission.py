"""Fit structural completion admission to actual source-view retrieval benefit."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def choose_admission_threshold(score,native_iou,completion_iou):
    if not len(score):raise ValueError('source benefit observations required')
    candidates=torch.cat([score.new_tensor([0.,1.]),score.unique()]).unique().sort().values
    utility=torch.where(score[None]>candidates[:,None],completion_iou[None],native_iou[None]).mean(1)
    # Ties prefer preserving native output, never arbitrary candidate replacement.
    best=torch.where(utility==utility.max())[0][-1]
    return float(candidates[best]),float(utility[best])


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'hard_source_scope_gate96';r=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==r['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False);field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    ep=root/'caption_region_evidence.pt';assert sha256_file(ep)==json.loads(ep.with_suffix('.json').read_text())['sha256'];e=torch.load(ep,weights_only=False)
    assert e['geometry_binding']==payload['geometry_binding'] and e['source_only'] and not e['benchmark_data_used']
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    assert e['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    with torch.no_grad():
        q=e['text_embeddings'].float().to(a.device);head=field.grounded_transfer;_,admitted,_=head.route(q)
        native_cosine=field.native_cosine(q);threshold=field.native_query_thresholds['text'](q);scale=field.readout_log_scale['text'].clamp(-4,6).exp()
        native=torch.sigmoid(scale*(native_cosine-threshold));support=head.support_overlap(field.latent,native_cosine>threshold,field.observed).max(1).values
        completion=head(q,field.latent,native_support=native_cosine>threshold,observed=field.observed)[0].sigmoid()
        responses=torch.cat([native,completion],1);h,w=e['masks'].shape[-2:];records=[];Q=len(q)
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);rec=auth['views'][vi];p=root/rec['relative_path'];assert sha256_file(p)==rec['sha256'];hit=torch.load(p,map_location=a.device,weights_only=False)
            gid=hit['gaussian_ids'].long();pid=hit['pixel_ids'].long();weight=hit['base_weights'].float()
            full=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight);keep=field.observed[gid];gid,pid,weight=gid[keep],pid[keep],weight[keep]
            mass=torch.zeros_like(full).scatter_add_(0,pid,weight);valid=(mass/full.clamp_min(1e-8)>.5)&(full>1e-8)
            mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,len(field.latent))).coalesce()
            rendered=torch.sparse.mm(mat,responses)>.5;rendered &= valid[:,None]
            targets={}
            for ri in torch.where(e['proposal_views']==vi)[0].tolist():
                for pi in e['records'][ri]['phrase_indices']:
                    if admitted[pi] or support[pi]<=0:continue
                    mask=e['masks'][ri].to(a.device).bool().flatten()
                    targets[pi]=targets.get(pi,torch.zeros_like(mask))|mask
            for pi,target in targets.items():
                # Keep the full target denominator, including unsupported positives.
                values=[]
                for col in [pi,Q+pi]:
                    prediction=rendered[:,col];values.append(float((prediction&target).sum()/(prediction|target).sum().clamp_min(1)))
                records.append(dict(phrase=e['phrases'][pi],phrase_index=pi,view=vi,score=float(support[pi]),native_iou=values[0],completion_iou=values[1]))
    unique=sorted({v['phrase_index'] for v in records});order=torch.tensor(unique)[torch.randperm(len(unique))];heldout=set(order[::5].tolist())
    train=[v for v in records if v['phrase_index'] not in heldout];validation=[v for v in records if v['phrase_index'] in heldout]
    if not train or not validation:raise ValueError('insufficient phrase-disjoint source validation')
    def tensors(rows):return [torch.tensor([v[k] for v in rows]) for k in ['score','native_iou','completion_iou']]
    minimum,_=choose_admission_threshold(*tensors(train))
    def measure(rows):
        score,native,completion=tensors(rows)
        return dict(observations=len(rows),phrases=len({v['phrase'] for v in rows}),native_iou=float(native.mean()),old_completion_iou=float(completion.mean()),gated_iou=float(torch.where(score>minimum,completion,native).mean()),accepted=int((score>minimum).sum()))
    arch=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],identity_selection=dict(payload['architecture']['grounded_transfer']['identity_selection'],minimum_support_iou=minimum)))
    out=root/'source_completion_admission96';out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch),out/'field.pt')
    report=dict(schema='completion_admission.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=r['field_sha256'],source_only=True,benchmark_data_used=False,minimum_support_iou=minimum,
        source_train=measure(train),source_validation=measure(validation),heldout_query_phrases=[e['phrases'][i] for i in sorted(heldout)],observations=records,
        all_parent_tensors_unchanged=True,validation_scope='query phrases held out of threshold selection only; base field and candidate extents saw source views',evidence_sha256=sha256_file(ep),
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,minimum,report['source_train'],report['source_validation'],flush=True)


if __name__=='__main__':main()
