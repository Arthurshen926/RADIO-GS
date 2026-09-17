"""Matched native/coarse-mask descriptor controls on the same new source regions."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    info=json.loads((root/'teachers.json').read_text());receipt=json.loads((root/'episodes.json').read_text())
    assert sha256_file(root/'episodes.pt')==receipt['sha256'];data=torch.load(root/'episodes.pt',weights_only=False)
    path=root/'caption_region_evidence.pt';er=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==er['sha256']
    evidence=torch.load(path,weights_only=False);assert evidence['geometry_binding']==data['geometry_binding'] and evidence['teacher_manifest_sha256']==data['teacher_manifest_sha256']
    parent=root/'full_source_query_threshold_control';report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False);assert payload['geometry_binding']==data['geometry_binding']
    trainviews={info['frames'].index(f) for f in info['train_frames']};train=torch.tensor([int(v) in trainviews for v in data['proposal_views']])
    oldkeys={(int(v),m.numpy().tobytes()) for v,m in zip(data['proposal_views'],data['masks'])}
    added=torch.tensor([i for i,(v,m) in enumerate(zip(evidence['proposal_views'],evidence['masks'])) if (int(v),m.numpy().tobytes()) not in oldkeys])
    assert len(added)>0
    oldmass=data['region_probability'][train].float().sum(0).to(a.device)
    weights=evidence['region_probability'][added].float().to(a.device);newmass=weights.sum(0)
    basis=payload['state_dict']['native_decoder.weight'].to(a.device);dim=basis.shape[1]
    old=payload['state_dict']['latent'][:,:dim].to(a.device);observed=payload['state_dict']['observed']
    for mode in ['native','coarse']:
        descriptors=F.normalize(evidence[mode+'_descriptors'][added].float().to(a.device),dim=-1)@basis
        fused=(oldmass[:,None]*old+weights.T@descriptors)/(oldmass+newmass)[:,None].clamp_min(1e-8)
        state={k:v.clone() for k,v in payload['state_dict'].items()};state['latent'][observed,:dim]=fused[observed.to(a.device)].cpu()
        out=root/('caption_'+mode+'_fusion_control');out.mkdir(exist_ok=False)
        torch.save(dict(payload,state_dict=state),out/'field.pt')
        r=dict(report,schema='caption_region_fusion_control.v1',field_sha256=sha256_file(out/'field.pt'),
            parent_field_sha256=report['field_sha256'],evidence_sha256=er['sha256'],added_regions=added.tolist(),descriptor_mode=mode,
            native_prefix_exactly_preserved=False,native_cosine_exactly_preserved=False,all_parent_tensors_unchanged=False,
            image_prompt_unchanged=False,prompt_unchanged=True,structural_state_exactly_preserved=True,
            basis_unchanged=True,observed_unchanged=True,inherited_calibration=True,benchmark_data_used=False,
            new_descriptor_projection_energy=float(descriptors.square().sum(-1).mean()),implementation_sha256=sha256_file(__file__))
        (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(a.scene,mode,len(added),flush=True)


if __name__=='__main__':main()
