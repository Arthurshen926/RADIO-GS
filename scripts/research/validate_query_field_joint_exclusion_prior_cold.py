"""Fresh field+query API; cached tensors are comparators, not inference inputs."""
import argparse,builtins,io,os
from pathlib import Path
import torch
import numpy as np
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder,query_text_joint_exclusion,_initial_weights,_prune
from scripts.research.run_query_field_joint_exclusion_prior import ROOT,read,write


def run(index):
    job=read(ROOT/'prediction_config.json')['jobs'][index]
    expected=torch.load(ROOT/'predictions'/job['name']/'predictions.pt',weights_only=True)
    queries=expected['native_query_vectors'].cuda();negatives=expected['generic_negative_vectors'].cuda()
    # Initialize the numerical JIT on synthetic data before the scene guard.
    # Its first-use plugin discovery scans installed package metadata, including
    # unrelated editable installs. No field/source/query values enter warmup.
    _initial_weights(np.array([0,1]),np.array([0]),np.array([.5]),np.array([True]))
    _prune(np.array([0,1],np.int32),np.array([0],np.int32),np.array([.5]),np.array([0]),np.array([True]),np.array([False]),np.array([True]),np.array([.5]),.1,.1001)
    allowed=Path(job['field']).resolve();attempts=[];originals=(builtins.open,io.open,os.open)
    def wrap(fn):
        def checked(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(path)).resolve()
                if ('output' in p.parts or str(p).startswith('/mnt/')) and not p.is_relative_to(allowed):
                    attempts.append(str(p));raise PermissionError('source access denied')
            return fn(path,*args,**kwargs)
        return checked
    builtins.open,io.open,os.open=[wrap(fn) for fn in originals]
    try:
        field,manifest=load_field(allowed,'cuda');decoder=JointExclusionDecoder(field)
        errors=dict(semantic=0.,member=0.);sets=True
        for qi,q in enumerate(queries):
            result=query_text_joint_exclusion(field,q,negatives,decoder=decoder,initializer="semantic")
            errors['semantic']=max(errors['semantic'],float((result.semantic_probability.cpu()-expected['scores']['joint_coverage'][:,qi]).abs().max()))
            errors['member']=max(errors['member'],float((result.membership_score.cpu()-expected['membership_by_control']['joint_coverage'][:,qi]).abs().max()))
            sets=sets and torch.equal(result.selected.cpu(),expected['selected_by_control']['joint_coverage'][:,qi])
        passed=sets and max(errors.values())<=2e-6 and not attempts
    finally:builtins.open,io.open,os.open=originals
    report=dict(passed=passed,scene=job['scene'],queries=len(queries),actual_sets_exact=sets,max_errors=errors,tolerance=2e-6,
        denied_attempts=attempts,python_guard_not_os_sandbox=True,numerical_runtime_initialized_on_synthetic_data_before_guard=True,source_raster_access=False,query_vectors_are_inputs=True,
        cached_response_is_comparator_not_input=True,query_parameter_fitting=False,field_sha256=manifest['files']['field.pt']['sha256'])
    write(ROOT/'validation'/('cold_'+job['scene']+'.json'),report);print(report,flush=True)
    if not passed:raise ValueError('cold native query gate failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);a=p.parse_args();torch.set_num_threads(4);run(a.index)
