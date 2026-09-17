"""Matched source-membership supervision using the original fixed vocabulary."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.training.region_visibility import normalize_visible_region_mass
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    out=root/'fixed_vocab_region_evidence.pt'
    if out.exists():raise FileExistsError(out)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    rawpath=root/'source_grounded_text.pt';raw=torch.load(rawpath,weights_only=False)
    rec=json.loads(rawpath.with_suffix('.json').read_text());assert sha256_file(rawpath)==rec['sha256']
    assert raw['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and raw['frames']==info['train_frames'] and not raw['benchmark_labels_used']
    n=auth['num_gaussians'];h,w=raw['probability'].shape[-2:];records=[];views=[];prob=[];seen=torch.zeros(len(info['frames']),n,dtype=torch.bool)
    for i,frame in enumerate(raw['frames']):
        vi=info['frames'].index(frame);masks=raw['probability'][i]>.5
        ids=torch.where((masks.flatten(1).sum(1)>=2)&(raw['scores'][i]>=.7))[0]
        r=auth['views'][vi];path=root/r['relative_path'];assert sha256_file(path)==r['sha256']
        hit=torch.load(path,map_location=a.device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float()
        mass=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight);weight=weight/mass[pid].clamp_min(1e-8)
        mapping=torch.sparse_coo_tensor(torch.stack([gid,pid]),weight,(n,h*w)).coalesce();elementmass=torch.zeros(n,device=a.device).scatter_add_(0,gid,weight)
        lifted,visible=normalize_visible_region_mass(torch.sparse.mm(mapping,masks[ids].float().flatten(1).to(a.device).T),elementmass)
        prob.append(lifted.T.cpu().half());seen[vi]=visible.cpu()
        for qi in ids.tolist():records.append(dict(frame=frame,phrase_indices=[qi]));views.append(vi)
    probability=torch.cat(prob);views=torch.tensor(views)
    parent=torch.load(root/'full_source_query_threshold_control/field.pt',weights_only=False);observed=parent['state_dict']['observed']
    groups=[]
    for qi,phrase in enumerate(raw['grounding_phrases']):
        ids=[j for j,r in enumerate(records) if qi in r['phrase_indices']];vs=sorted({int(views[j]) for j in ids})
        if len(vs)<2:continue
        numerator=torch.zeros(n);denominator=torch.zeros(n)
        for vi in vs:
            js=[j for j in ids if int(views[j])==vi];numerator+=probability[js].float().amax(0);denominator+=seen[vi]
        membership=numerator/denominator.clamp_min(1)
        if int(((membership>.5)&observed).sum())<20:continue
        groups.append(dict(phrase=phrase,source_query_index=qi,views=vs,regions=ids))
    text=NativeThreeQueryEncoder(parent['query_encoder'],a.device).encode_text(raw['grounding_phrases']).cpu().half()
    bundle=dict(schema='fixed_vocab_region_evidence.v1',source_only=True,benchmark_data_used=False,
        source_grounding_sha256=rec['sha256'],teacher_manifest_sha256=sha256_file(root/'teachers.json'),geometry_binding=parent['geometry_binding'],
        train_frames=info['train_frames'],region_probability=probability,seen=seen,proposal_views=views,
        records=records,phrases=raw['grounding_phrases'],text_embeddings=text,implementation_sha256=sha256_file(__file__))
    torch.save(bundle,out);digest=sha256_file(out)
    out.with_suffix('.json').write_text(json.dumps(dict(sha256=digest,regions=len(probability),implementation_sha256=bundle['implementation_sha256']),indent=2)+'\n')
    (root/'fixed_vocab_membership_groups.json').write_text(json.dumps(dict(evidence_sha256=digest,groups=groups),indent=2)+'\n')
    print(a.scene,len(groups),len(probability),flush=True)


if __name__=='__main__':main()
