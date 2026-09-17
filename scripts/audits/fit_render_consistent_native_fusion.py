"""Source-only native fusion control: invert source rendering, no query training.

Existing structural heads are deliberately not used with changed coordinates.
Exports from the native-only reference, preserving its calibration and support.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.render_consistent_fusion import fit_render_consistent_features
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    out=root/'render_consistent_native_control';assert not out.exists()
    bindings={}
    def load(name,digest):
        path=root/name;assert sha256_file(path)==digest;bindings[name]=digest
        return torch.load(path,weights_only=False)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    data=load('episodes.pt',json.loads((root/'episodes.json').read_text())['sha256']);assert data['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    tr=info['teacher_files']['siglip'];teacher=load(tr['path'],tr['sha256'])
    parent='full_source_query_threshold_control';pr=json.loads((root/parent/'report.json').read_text());payload=load(parent+'/field.pt',pr['field_sha256']);assert payload['geometry_binding']==data['geometry_binding']
    assert not payload['architecture'].get('grounded_semantic_transfer')
    sd=payload['state_dict'];observed=sd['observed'].to(a.device);initial=sd['latent'][:,:64].to(a.device);basis=sd['native_decoder.weight'].to(a.device)
    desc=F.normalize(teacher['descriptors'].float().to(a.device),dim=-1);masks=data['masks'].flatten(1).float().to(a.device);pixels=masks.shape[1];n=len(initial)
    allids=[];allvalues=[];targets=[];rows=0
    with torch.no_grad():
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);ids=torch.where(data['proposal_views']==vi)[0]
            pixelnum=masks[ids].T@desc[ids];covered=masks[ids].sum(0)>0
            target=F.normalize(pixelnum,dim=-1)@basis
            rec=auth['views'][vi];hit=load(rec['relative_path'],rec['sha256'])
            gid,pid,w=[hit[k].to(a.device) for k in ['gaussian_ids','pixel_ids','base_weights']];gid=gid.long();pid=pid.long();w=w.float()
            fullmass=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w)
            w=w*observed[gid];mass=torch.zeros(pixels,device=a.device).scatter_add_(0,pid,w)
            valid=covered&(mass/fullmass.clamp_min(1e-8)>.5)&(mass>1e-8)
            rowmap=torch.full((pixels,),-1,device=a.device,dtype=torch.long);rowmap[valid]=torch.arange(int(valid.sum()),device=a.device)+rows
            keep=valid[pid]&(w>0)
            allids.append(torch.stack([rowmap[pid[keep]],gid[keep]]));allvalues.append(w[keep]/mass[pid[keep]])
            targets.append(target[valid]);rows+=int(valid.sum())
        operator=torch.sparse_coo_tensor(torch.cat(allids,1),torch.cat(allvalues),(rows,n)).coalesce();target=torch.cat(targets)
        estimate,metrics=fit_render_consistent_features(operator,target,initial)
        assert torch.isfinite(estimate).all() and metrics['source_mse_after']<=metrics['source_mse_before']
        state={k:v.clone() for k,v in sd.items()};state['latent'][:,:64]=estimate.cpu()
        for k,v in sd.items():
            if k!='latent':assert torch.equal(v,state[k])
        assert torch.equal(sd['latent'][:,64:],state['latent'][:,64:])
    out.mkdir();torch.save(dict(payload,state_dict=state),out/'field.pt')
    report=dict(schema='render_consistent_native_fusion.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=pr['field_sha256'],
        source_only=True,benchmark_data_used=False,metrics=metrics,source_pixels=rows,prior_weight=.1,max_iterations=80,
        teacher_target='normalized full-native mean of covering original source region descriptors, projected with unchanged basis',
        objective='source pixel squared error plus 0.1 diag(A.T A) prior to original prefix; cached observed-normalized contribution operator',
        support_and_basis_unchanged=True,structure_suffix_unchanged=True,calibration_inherited=True,native_prefix_exactly_preserved=False,
        old_extent_heads_used=False,inputs=bindings,teacher_manifest_sha256=sha256_file(root/'teachers.json'),authority_sha256=info['authority_sha256'],
        implementation_sha256=sha256_file(__file__),solver_sha256=sha256_file('radio_gs/training/render_consistent_fusion.py'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,metrics,flush=True)


if __name__=='__main__':main()
