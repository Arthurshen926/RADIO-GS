"""Full historical region decision parity on label-free validation-room seeds.

Construction associations are opened by this offline validator only. No claim
that these mesh features are the current Gaussian field's deployed features.
"""
import builtins
import json
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
from radio_gs.querying.conditional_region_membership import ConditionalRegionMembership
from radio_gs.query_field.region_quality import RegionQuality
from radio_gs.query_field.region_admission import admit_regions
from radio_gs.query_field.storage import sha256
from scripts.research.validate_query_field_region_quality import SOURCE, OUT


def main():
    out=OUT/'admission_audit.json'
    if out.exists():raise FileExistsError(out)
    quality=RegionQuality(torch.load(OUT/'quality.pt',weights_only=True))
    design=json.loads((SOURCE/'design.json').read_text());scenes={};receipts=[]
    for scene in design['validation']:
        folder=SOURCE/'maps'/scene
        names=['membership.npz','visibility.npz','region_offsets.npy','mesh_xyz.npy','appearance_descriptors.npy']
        receipts.extend(dict(path=str(folder/n),sha256=sha256(folder/n)) for n in names)
        b=sparse.load_npz(folder/'membership.npz');v=sparse.load_npz(folder/'visibility.npz')
        frames=np.repeat(np.arange(v.shape[1]),np.diff(np.load(folder/'region_offsets.npy')))
        original=ConditionalRegionMembership(b,v,frames,np.load(folder/'mesh_xyz.npy'),np.load(folder/'appearance_descriptors.npy'),SOURCE/'conditional_membership.joblib')
        original.model['estimator'].n_jobs=1
        available=np.flatnonzero(original.features.area>0)
        seeds=available[np.unique(np.linspace(0,len(available)-1,min(8,len(available)),dtype=int))]
        rows=[]
        for seed in seeds:
            identity=np.zeros(len(frames));identity[seed]=1
            _,expected,detail=original.infer(identity,np.zeros(b.shape[0]))
            ids,x,shared=original.features.pairs(seed)
            args=dict(seed=int(seed),candidates=torch.from_numpy(ids),intersection=torch.from_numpy(shared),
                      area=torch.from_numpy(original.features.area),frames=torch.from_numpy(frames),view_count=v.shape[1])
            def denied(*a,**k):raise AssertionError('candidate runtime file read')
            saved=builtins.open
            try:
                builtins.open=denied
                result=admit_regions(quality,quality(torch.from_numpy(x)),**args)
            finally:builtins.open=saved
            assert (result is not None)==detail['applied']
            if result is not None:
                weights,confidence,selected=result
                assert selected.tolist()==[r['region'] for r in detail['admitted']]
                for r in detail['admitted']:
                    assert weights[r['region']].item()==float(np.float32(r['precision']))
                    assert confidence[frames[r['region']]].item()==float(np.float32(r['precision']))
                positive=np.asarray(b@weights.numpy()).ravel();known=np.asarray(v@confidence.numpy()).ravel()
                support=np.divide(positive,known,out=np.zeros_like(positive),where=known>0).clip(0,1)
                assert np.array_equal(support>.5,expected)
            rows.append(dict(seed=int(seed),pairs=len(ids),applied=detail['applied'],selected_exact=True,
                             admitted_regions=[] if result is None else result[2].tolist()))
        scenes[scene]=rows;print(scene,'seeds',len(rows),'applied',sum(r['applied'] for r in rows),flush=True)
    report=dict(passed=True,scenes=scenes,seeds=sum(len(v) for v in scenes.values()),
        seed_rule='8 evenly spaced nonempty region IDs per room; no labels or native text used to choose seeds',
        matched='original non-anchor conditional region admission, per-view choices, float32 weights and final majority member sets',
        python_open_denied_during_candidate=True,os_sandbox=False,benchmark_labels_opened=False,
        original_feature_generator_reused=True,gaussian_feature_transfer=False,anchor_path_verified=False,
        default_promoted=False,input_files=receipts,quality_export_sha256=sha256(OUT/'quality.pt'),
        code_sha256=sha256('radio_gs/query_field/region_admission.py'),validator_sha256=sha256(__file__))
    text=json.dumps(report,indent=2)+'\n';out.write_text(text)
    Path('docs/experiments/2026-10-11-region-admission-audit.json').write_text(text)

if __name__=='__main__':torch.set_num_threads(4);main()
