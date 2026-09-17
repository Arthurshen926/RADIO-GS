"""Bounded control: remove untrained channel-order weighting from prompt affinity.

A shared scalar gate cancels under feature normalization. Geometry, all learned
field values, text/image paths, seed snapping and decision rule are unchanged.
No target labels or source validation selection are used to choose this control.
"""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='joint_supervised_image_residual');p.add_argument('--name',default='uniform_structural_affinity');a=p.parse_args();torch.set_num_threads(4)
    src=a.root/a.parent;receipt=json.loads((src/'report.json').read_text());assert sha256_file(src/'field.pt')==receipt['field_sha256']
    payload=torch.load(src/'field.pt',map_location='cpu',weights_only=False);sd=payload['state_dict'];assert payload['architecture']['scale_affinity'] and not payload['architecture']['scale_affinity'].get('full_prompt_features',False)
    key='affinity.cutoffs';old=sd[key].clone();assert old.numel()==sd['latent'].shape[1]-payload['architecture']['native_semantic_dim']
    # All scales retain a common nonzero scalar gate; normalization cancels it.
    sd[key]=torch.zeros_like(old)
    from radio_gs.models.scale_affinity_readout import ScaleAffinityReadout
    kwargs=payload['architecture']['scale_affinity'];reader=ScaleAffinityReadout(old.numel(),**kwargs);reader.load_state_dict({k.removeprefix('affinity.'):v for k,v in sd.items() if k.startswith('affinity.')});scale=reader.scales[reader.default_scale]
    rows=torch.where(sd['observed'])[0];generator=torch.Generator().manual_seed(42);rows=rows[torch.randperm(len(rows),generator=generator)[:2048]];z=sd['latent'][rows,payload['architecture']['native_semantic_dim']:];perm=torch.randperm(z.shape[1],generator=generator)
    with torch.no_grad():
        reader.cutoffs.copy_(old);old_logits=reader.logits_at_scale(z,0,1,scale);permuted=reader.logits_at_scale(z[:,perm],0,1,scale);legacy_delta=float((old_logits.sigmoid()-permuted.sigmoid()).abs().max())
        reader.cutoffs.zero_();new=reader.logits_at_scale(z,0,1,scale);newperm=reader.logits_at_scale(z[:,perm],0,1,scale);new_delta=float((new.sigmoid()-newperm.sigmoid()).abs().max());torch.testing.assert_close(new.sigmoid(),newperm.sigmoid(),atol=2e-6,rtol=2e-5)
    out=a.root/a.name;out.mkdir(exist_ok=False);torch.save(payload,out/'field.pt');shutil.copy2(__file__,out/'source.py')
    r=dict(receipt,schema='uniform_structural_affinity_control.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=receipt['field_sha256'],implementation_sha256=sha256_file(__file__),changed_state_keys=[key],all_other_state_exact=True,benchmark_data_used=False,source_validation_selection=False,physical_scale_conditioning=False,control='constant scalar gate cancels in normalized structural affinity; preserves native text/image, latent, geometry and snapping',channel_permutation_probe=dict(elements=len(rows),random_seed=42,legacy_max_probability_change=legacy_delta,uniform_max_probability_change=new_delta),original_cutoffs=old.tolist())
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(r['channel_permutation_probe'])
if __name__=='__main__':main()
