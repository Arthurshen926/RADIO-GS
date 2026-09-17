"""Verify original raw source evidence survives adding mapping views."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('paper/artifacts/pose_shared_source_evidence_20260917.json'));a=p.parse_args();torch.set_num_threads(4);out=a.output;assert not out.exists();result=dict(implementation_sha256=sha256_file(__file__),scenes={},benchmark_data_used=False)
for scene in ('figurines','ramen'):
    roots=[Path('output')/d/scene/'raw_field' for d in ('lerf_ropefix_multiscale_pipeline_20260917','lerf_pose_covered_pipeline_20260917')];bundles=[];bindings={}
    for root in roots:
        info=json.loads((root/'teachers.json').read_text());bindings[str(root/'teachers.json')]=sha256_file(root/'teachers.json');loaded={}
        for kind in ('sam','siglip'):
            rec=info['teacher_files'][kind];p=root/rec['path'];assert sha256_file(p)==rec['sha256'];loaded[kind]=torch.load(p,map_location='cpu',weights_only=False)
        bundles.append((info,loaded))
    (old,ox),(new,nx)=bundles;rows=[];assert ox['siglip']['queries']==nx['siglip']['queries']
    for f in old['frames']:
        ids=[]
        for info,x in bundles:ids.append(torch.where(x['sam']['view_indices']==info['frames'].index(f))[0])
        a,b=ids;assert torch.equal(ox['sam']['native_proposal_indices'][a],nx['sam']['native_proposal_indices'][b]);assert torch.equal(ox['sam']['masks'][a],nx['sam']['masks'][b])
        delta=float((ox['siglip']['descriptors'][a].float()-nx['siglip']['descriptors'][b].float()).abs().max());rows.append(dict(frame=f,regions=len(a),mask_and_native_ids_exact=True,descriptor_max_difference=delta,descriptor_min_cosine=float(torch.nn.functional.cosine_similarity(ox['siglip']['descriptors'][a].float(),nx['siglip']['descriptors'][b].float()).min())))
    result['scenes'][scene]=dict(inputs=bindings,same_fixed_grounding_vocabulary=True,rows=rows)
out.write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,out.with_suffix('.source.py'));print({s:max(x['descriptor_max_difference'] for x in r['rows']) for s,r in result['scenes'].items()})
