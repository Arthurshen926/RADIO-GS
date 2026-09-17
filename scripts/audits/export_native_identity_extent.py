"""Pool exact native field identity within frozen learned extents.

Stores only one global native moment per candidate; no new per-element map.
An unnormalized mean of unit element features reproduces mean element cosine.
Normalizing that mean again would change its scale and invalidate the native
threshold, so it is deliberately not normalized.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'source_hierarchical_candidates96';receipt=json.loads((parent/'report.json').read_text())
    assert sha256_file(parent/'field.pt')==receipt['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
    field=ElementRetrievalField(**payload['architecture']).to(a.device);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    head=field.grounded_transfer;R=head.num_proposals;D=field.native_decoder.out_features
    moments=torch.zeros(R,D,device=a.device);mass=torch.zeros(R,device=a.device)
    q=F.normalize(torch.randn(3,D,device=a.device),dim=-1);direct=torch.zeros(R,3,device=a.device)
    with torch.no_grad():
        for rows in torch.arange(len(field.latent),device=a.device).split(4096):
            z=field.latent[rows];features=head.features(z)
            mask=((head.decoder(features)+head.residual(features))>.5)&field.observed[rows,None]
            native=F.normalize(field.native_decoder(z[:,:field.native_decoder.in_features]),dim=-1)
            weights=mask.float();moments+=weights.T@native;mass+=weights.sum(0);direct+=weights.T@(native@q.T)
        moments/=mass[:,None].clamp_min(1);direct/=mass[:,None].clamp_min(1)
        max_error=float((moments@q.T-direct).abs().max());assert max_error<1e-6
    for name,config in [('native_identity_restricted96',dict(restrict_concepts=True,require_native_support=False)),
                        ('native_identity_global96',dict(restrict_concepts=False,require_native_support=True))]:
        architecture=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],identity_selection=config))
        new=ElementRetrievalField(**architecture);missing=new.load_state_dict(payload['state_dict'],strict=False)
        assert missing.missing_keys==['grounded_transfer.native_extent_means'] and not missing.unexpected_keys
        new.grounded_transfer.native_extent_means.copy_(moments.cpu())
        state=new.state_dict()
        for k,v in payload['state_dict'].items():assert torch.equal(v,state[k]),k
        out=root/name;out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
        report=dict(receipt,schema='native_identity_extent.v1',parent_field_sha256=receipt['field_sha256'],field_sha256=sha256_file(out/'field.pt'),
            source_only=True,benchmark_data_used=False,all_parent_tensors_unchanged=True,identity_policy=config,
            moment_contract='mean normalized native decoded field feature inside thresholded candidate extent; observed elements only',
            empty_candidates=int((mass==0).sum()),candidate_element_mass=mass.cpu().tolist(),mean_cosine_max_error=max_error,
            implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(a.scene,R,max_error,flush=True)


if __name__=='__main__':main()
