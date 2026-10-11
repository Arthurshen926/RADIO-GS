"""Function-preservation audit on historical independent calibration/validation rooms.

No benchmark samples, training, hyperparameter choice or Gaussian deployment.
The same 16-dimensional original feature arrays enter both implementations.
"""
import builtins
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from radio_gs.query_field.region_quality import RegionQuality,compile_region_quality,TARGETS
from radio_gs.query_field.storage import sha256

SOURCE=Path('output/learned_instance_additions')
OUT=Path('output/query_field_20261011/region_quality_transfer_v1')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    checkpoint=joblib.load(SOURCE/'conditional_membership.joblib')
    design=json.loads((SOURCE/'design.json').read_text())
    splits=[set(design[k]) for k in ['fit','calibration','validation']]
    assert all(not a&b for i,a in enumerate(splits) for b in splits[i+1:])
    for k in ['fit','calibration','validation']:assert checkpoint[k]==design[k]
    state=compile_region_quality(checkpoint);torch.save(state,OUT/'quality.pt')
    compiled=RegionQuality(torch.load(OUT/'quality.pt',weights_only=True)).eval()
    reports={};records=[];arrays=[];historical=[];reference=[]
    for split in ['calibration','validation']:
        for scene in design[split]:
            path=SOURCE/'training'/f'{scene}.npz';data=np.load(path)
            x=data['x'];records.append(dict(path=str(path),sha256=sha256(path)))
            checkpoint['estimator'].n_jobs=4;parallel=checkpoint['estimator'].predict(x)
            checkpoint['estimator'].n_jobs=1;canonical=checkpoint['estimator'].predict(x)
            arrays.append((scene,split,torch.from_numpy(x),data['y'],data['addition']))
            historical.append(parallel);reference.append(canonical)
    def denied(*args,**kwargs):raise AssertionError('runtime source read')
    original=builtins.open
    try:
        builtins.open=denied
        outputs=[compiled(x).numpy() for _,_,x,_,_ in arrays]
    finally:builtins.open=original
    for (scene,split,x,truth,addition),actual,expected,parallel in zip(arrays,outputs,reference,historical):
        np.testing.assert_array_equal(actual,expected)
        threshold=checkpoint['addition_threshold']
        admission=actual[:,2]>=threshold
        assert np.array_equal(admission,expected[:,2]>=threshold)
        allowed=admission & (addition[:,1]>0)
        reports[scene]=dict(split=split,pairs=len(x),prediction_exact_canonical=True,
            admission_exact_canonical=True,max_parallel_value_difference=float(np.max(np.abs(actual-parallel))),
            parallel_admission_disagreements=int((admission!=(parallel[:,2]>=threshold)).sum()),
            admitted_with_known_additions=int(allowed.sum()),
            known_added_purity=float(addition[allowed,0].sum()/max(addition[allowed,1].sum(),1)),
            mean_absolute_error=np.abs(actual-truth).mean(0).tolist())
    report=dict(passed=True,targets=TARGETS,scenes=reports,rows=sum(r['pairs'] for r in reports.values()),
        addition_threshold=checkpoint['addition_threshold'],tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()),
        trees=len(state['roots']),nodes=len(state['feature']),input_features_unchanged=True,
        inference_python_open_denied=True,os_sandbox=False,training=False,benchmark_data_used=False,
        gaussian_feature_semantics_verified=False,field_integration=False,default_promoted=False,
        full_region_selection_equivalence=False,
        reference='sklearn original forest, canonical n_jobs=1 tree-order accumulation; n_jobs=4 also checked separately',
        source=dict(path=str(SOURCE/'conditional_membership.joblib'),sha256=sha256(SOURCE/'conditional_membership.joblib')),
        design=dict(path=str(SOURCE/'design.json'),sha256=sha256(SOURCE/'design.json')),inputs=records,
        export_sha256=sha256(OUT/'quality.pt'),code_sha256=sha256('radio_gs/query_field/region_quality.py'),validator_sha256=sha256(__file__))
    text=json.dumps(report,indent=2)+'\n';(OUT/'audit.json').write_text(text)
    Path('docs/experiments/2026-10-11-region-quality-transfer-audit.json').write_text(text)
    print(text,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
