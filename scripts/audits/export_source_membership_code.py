"""Compress source phrase membership into the existing 32 structural coordinates."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--balance-groups',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    path=root/'caption_region_evidence.pt';receipt=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==receipt['sha256']
    evidence=torch.load(path,weights_only=False);info=json.loads((root/'teachers.json').read_text())
    assert evidence['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and evidence['train_frames']==info['train_frames']
    parent=root/'scale_affinity_v1_direct_control';report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False);assert payload['geometry_binding']==evidence['geometry_binding']
    observed=payload['state_dict']['observed'].to(a.device);n=len(observed)
    probability=evidence['region_probability'].float().to(a.device);seen=evidence['seen'].to(a.device)
    groups=[];indices=[];images=[];descriptions=[]
    for qi,phrase in enumerate(evidence['phrases']):
        ids=[i for i,r in enumerate(evidence['records']) if qi in r['phrase_indices']]
        views=sorted({int(evidence['proposal_views'][i]) for i in ids})
        if len(views)<2:continue
        numerator=torch.zeros(n,device=a.device);denominator=torch.zeros_like(numerator)
        for vi in views:
            members=[i for i in ids if int(evidence['proposal_views'][i])==vi]
            numerator+=probability[members].amax(0)
            denominator+=seen[vi]
        membership=(numerator/denominator.clamp_min(1))*observed
        if int((membership>.5).sum())<20:continue
        groups.append(membership);indices.append(qi)
        images.append(F.normalize(evidence['native_descriptors'][ids].float(),dim=-1).mean(0))
        descriptions.append(dict(phrase=phrase,source_query_index=qi,views=views,regions=ids))
    matrix=torch.stack(groups);structural_dim=payload['architecture']['latent_dim']-payload['architecture']['native_semantic_dim']
    # Uncentered exact PCA of source memberships, no benchmark information.
    row_scale=matrix.norm(dim=1).clamp_min(1e-8) if a.balance_groups else torch.ones(len(matrix),device=a.device)
    normalized=matrix/row_scale[:,None]
    eigval,eigvec=torch.linalg.eigh(normalized@normalized.T)
    decoder=eigvec[:,-min(structural_dim,len(matrix)):]
    code=normalized.T@decoder
    decoder=decoder*row_scale[:,None]
    if code.shape[1]<structural_dim:
        pad=structural_dim-code.shape[1];code=F.pad(code,(0,pad));decoder=F.pad(decoder,(0,pad))
    reconstructed=code@decoder.T
    target=matrix.T>.5;pred=reconstructed>.5
    iou=((target&pred).sum(0)/(target|pred).sum(0).clamp_min(1)).mean()
    architecture=dict(payload['architecture'],source_membership_code={'num_groups':len(groups),'query_dim':evidence['text_embeddings'].shape[1]})
    field=ElementRetrievalField(**architecture);missing=field.load_state_dict(payload['state_dict'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('source_membership.') for k in missing.missing_keys)
    dim=payload['architecture']['native_semantic_dim']
    with torch.no_grad():
        field.latent[:,dim:].copy_(code.cpu());field.source_membership.decoder.weight.copy_(decoder.cpu())
        field.source_membership.text_prototypes.copy_(evidence['text_embeddings'][indices].float())
        field.source_membership.image_prototypes.copy_(F.normalize(torch.stack(images),dim=-1))
    assert torch.equal(field.latent[:,:dim],payload['state_dict']['latent'][:,:dim])
    state=field.state_dict();out=root/('source_membership_balanced32' if a.balance_groups else 'source_membership_code32');out.mkdir(exist_ok=False)
    torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
    r=dict(report,schema='source_membership_code.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=report['field_sha256'],
        evidence_sha256=receipt['sha256'],source_only=True,benchmark_data_used=False,groups=descriptions,structural_dimension=structural_dim,
        balance_groups=a.balance_groups,source_compression_mean_iou=float(iou),source_compression_energy=float(eigval[-structural_dim:].sum()/eigval.sum()),
        native_prefix_exactly_preserved=True,native_cosine_exactly_preserved=True,native_text_image_max_error=None,
        structural_training=False,structural_construction='analytical PCA of source class-union memberships; replaces previous structure, not an extra per-element field',
        query_contract='nearest native text/image prototype selects one source phrase union; signed prompts select compatible source union; absence and novel ungrounded concepts not validated',
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/source_membership_code.py'))
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(a.scene,len(groups),float(iou),r['source_compression_energy'],flush=True)


if __name__=='__main__':main()
