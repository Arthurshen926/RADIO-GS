"""Replay the actual v10 LERF production response stage, including pixel phases.

This uses the retained production artifacts, rather than the older research
control artifacts. Both retain the original pixel phases. No evaluator labels,
new response tensors or copied teacher/model checkpoints are saved.
"""
import gc
import json
from pathlib import Path
import torch

from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.pipeline.response import fuse_admitted_evidence
from radio_gs.query_field.bounded_experiment import BoundedExperiment,tree_bytes
from radio_gs.querying.response_study_adapters import SYSTEM,read,load
from scripts.research.fit_consolidated_response import iter_source_response_views
from scripts.research.replay_tmm_response import STUDY,POLICY,sha


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    run=BoundedExperiment(STUDY,'runtime_core_replay',POLICY)
    paths=['radio_gs/pipeline/response.py','radio_gs/querying/response_refinement.py',
        'scripts/research/fit_consolidated_response.py','scripts/research/replay_tmm_runtime_response.py']
    run.json('source_snapshot.json',{p:dict(sha256=sha(p),text=Path(p).read_text()) for p in paths})
    rows=[]
    for name in ['figurines','ramen','teatime','waldo_kitchen']:
        root=SYSTEM/'lerf'/name/'query/text/response';fit=read(root/'fit.json')
        inputs=read(fit['inputs']);scene=read(inputs['scene']);data=load(fit['readout'])
        if fit['solver']!='backprojection' or fit['finite_evidence'] or scene.get('instance_evidence_reliability'):
            raise ValueError('replay must preserve declared production policy')
        if fit['raster']!=[192,256] or fit['sample']!='one of four interleaved pixel phases per view; same phase as subset solver':
            raise ValueError('retained production sampling differs')
        b=data['semantic_geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda')
        gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256'])
        prior=data['scores'].cuda();observed=data['semantic_observed'].cuda()
        old_mass=torch.zeros_like(prior);old_numerator=torch.zeros_like(prior);count=0
        def observations():
            nonlocal count
            for op,target,pos,neg,native,confidence in iter_source_response_views(inputs,scene,data,geo,gp,observed,prior):
                # The shared iterator already applies the original i % 4 phase.
                if op.shape[0]!=192*256//4:raise ValueError('pixel phase sampling changed')
                known=(pos|(neg&~native)).float()*confidence[None]
                old_mass.add_(torch.sparse.mm(op.T,known))
                old_numerator.add_(torch.sparse.mm(op.T,known*target))
                count+=1
                yield op,target,known
        values,mass=fuse_admitted_evidence(prior,observations())
        old=torch.where(old_mass>1e-8,old_numerator/old_mass.clamp_min(1e-8),prior)
        fresh_error=float((values-old).abs().max());values[~observed]=0
        expected=load(root/'native_protected/typed_readout.pt')
        retained_error=float((values.cpu()-expected['scores']).abs().max())
        crossings=int(((values.cpu()>.5)!=(expected['scores']>.5)).sum())
        selection_same=torch.equal(data['selection_scores'],expected['selection_scores'])
        passed=fresh_error<=1e-6 and retained_error<=1e-6 and crossings==0 and selection_same
        row=dict(scene=name,source_views=count,query_columns=len(data['texts']),
            fresh_old_core_max_error=fresh_error,retained_response_max_error=retained_error,
            response_half_threshold_crossings=crossings,selection_exact=selection_same,passed=passed,
            fit_sha256=sha(root/'fit.json'),initial_readout_sha256=sha(fit['readout']),
            retained_readout_sha256=sha(root/'native_protected/typed_readout.pt'))
        run.json(name+'.json',row);rows.append(row);print(json.dumps(row),flush=True)
        if not passed:raise AssertionError('production-stage compatibility failed')
        del data,prior,values,old,expected,geo,old_mass,old_numerator,mass
        gc.collect();torch.cuda.empty_cache()
    run.json('summary.json',dict(complete=True,rows=rows,ground_truth_read=False,
        production_pixel_phase_sampling=True,pipeline_entry_calls_shared_core=True,
        max_retained_error=max(r['retained_response_max_error'] for r in rows),
        total_threshold_crossings=sum(r['response_half_threshold_crossings'] for r in rows),
        unaffected_inference_families=['nvos','scannet','uqis'],storage_bytes=tree_bytes(STUDY)))


if __name__=='__main__':main()
