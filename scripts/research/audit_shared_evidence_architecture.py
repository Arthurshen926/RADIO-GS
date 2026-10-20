"""Read-only route inventory; distinguishes code sharing from planned mechanisms."""
import argparse
import hashlib
import json
from pathlib import Path

from radio_gs.pipeline.runtime import PipelineConfig, build_plan

ROOT=Path(__file__).resolve().parents[2]

CATALOG=[
    dict(stage='build',module='radio_gs/pipeline/observation_field.py',status='partly_shared',
         scope='Geometry renderer stays specialized; shared region contribution algebra and optional separate visibility',
         caveat='CPU algebra checked; new GPU builds and visibility cost remain unverified.'),
    dict(stage='build',module='radio_gs/evidence_core/gaussian_observations.py',status='shared_implementation',
         scope='Overlap-aware Gaussian contribution and separate unconditioned geometric mass',
         caveat='Distinct from equal-view registered normalization; no historical visibility is inferred.'),
    dict(stage='store',module='radio_gs/evidence_core/provenance.py',status='shared_implementation',
         scope='Content-bound region/view metadata overlays; exact native descriptor and replacement transfer lineage',
         caveat='Source masks and replacement encoder are not replayed; original fields remain immutable.'),
    dict(stage='build',module='radio_gs/pipeline/registered_observation_map.py',status='partly_shared',
         scope='RGB-D registration stays specialized; shared view-balanced region normalization is opt-in',
         caveat='Not equivalent to Gaussian contribution-mass normalization.'),
    dict(stage='store',module='radio_gs/evidence_core/store.py',status='shared_interface_and_validation',
         scope='Read-only compiled and registered adapters; shared region/view layout validation',
         caveat='Compiled support is not geometric visibility; missing provenance remains missing.'),
    dict(stage='query',module='radio_gs/evidence_core/query_contracts.py',status='shared_implementation',
         scope='Public text/image/2D-point/3D-point permissions; crop path and hash binding',
         caveat='NVOS signed reference and authorized target-RGB contract is separate, not converted to crop-only.'),
    dict(stage='query',module='radio_gs/querying/reference_admission.py',status='task_specific',
         scope='N14 signed reference repair decision',
         caveat='Both frozen policies integrated as opt-in profiles; fresh GPU migration parity remains pending.'),
    dict(stage='query',module='radio_gs/pipeline/reference_completion.py',status='task_specific_on_shared_core',
         scope='Fresh source-only initializer, complete N14 worker, common native-ray writeback',
         caveat='Explicit signed prompts and target RGB; inherited selected-set output is separately identified.'),
    dict(stage='evidence',module='radio_gs/evidence_core/core.py',status='shared_implementation',
         scope='Signed domains, sparse transport, view grouping and scope reconstruction',
         caveat='Identity decisions and task weight policies still originate in typed providers.'),
    dict(stage='readout',module='radio_gs/querying/response_refinement.py',status='task_specific_on_shared_core',
         scope='Continuous semantic consensus',caveat='No common-loss claim.'),
    dict(stage='readout',module='radio_gs/pipeline/subset.py',status='task_specific_on_shared_core',
         scope='Actual selected-Gaussian subset rendering',caveat='Only sparse transport is CPU checked; renderer needs GPU end-to-end validation.'),
    dict(stage='readout',module='radio_gs/querying/native_observation_prior.py',status='task_specific_on_shared_core',
         scope='Region menu competition precedes aggregation',caveat='Single-SigLIP2 replacement is a separate unimplemented profile.'),
    dict(stage='readout',module='radio_gs/querying/response_reconstruction.py',status='shared_implementation',
         scope='Admitted observation diagonal/Huber response formation',caveat='Does not infer identity or negative permissions.'),
    dict(stage='readout',module='radio_gs/querying/image_conditioned_response.py',status='task_specific_on_shared_core',
         scope='N14 uses the shared numerical solver for authorized image writeback',
         caveat='Numerical backend sharing alone does not make N14 a fresh unified recipe.'),
    dict(stage='readout',module='radio_gs/querying/direct_anchor_membership.py',status='task_specific_on_shared_core',
         scope='Complete trained membership; optional 8192-row feature blocks',
         caveat='Same checkpoint, feature order, threshold and complete carrier; training supervision remains disclosed.'),
    dict(stage='evaluation',module='radio_gs/pipeline/cohort.py',status='shared_orchestration_typed_evaluators',
         scope='All predictions sealed before labels; separate ScanNet internal10/paper8',
         caveat='No full new benchmark scores yet.'),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',default='configs/pipeline/system_v10.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    recipe=json.loads((ROOT/args.recipe).read_text());rows=[]
    for job in recipe['jobs']:
        options={key:job[key] for key in ['source_evidence','reference_tracks','scope_transfer','instance_scope','reference_completion'] if key in job}
        plan=build_plan(PipelineConfig(scene=job['scene'],queries=job['queries'],
            output=str(ROOT/'tmp/not-executed-architecture-audit'/job['benchmark']/job['name']),
            access=job.get('access','source-assisted'),**options))
        scene=json.loads((ROOT/job['scene']).read_text())
        rows.append(dict(benchmark=job['benchmark'],name=job['name'],scene=job['scene'],
            scene_sha256=hashlib.sha256((ROOT/job['scene']).read_bytes()).hexdigest(),
            region_provenance_overlay=scene.get('native_observation_provenance'),
            commands=plan['commands'],commands_executed=False,
            source_rgb_allowed=plan.get('source_rgb_allowed',False),
            target_rgb_allowed=plan.get('target_rgb_allowed',False),
            readout=plan.get('readout'),runtime_provider=plan.get('runtime_provider','typed_source_pipeline')))
    modules=[]
    for item in CATALOG:
        path=ROOT/item['module']
        modules.append(dict(item,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(schema='radiogs.architecture_inventory.v1',
        status='source_and_plan_audit_not_end_to_end_execution',modules=modules,jobs=rows,
        shared_build_complete=False,core_algorithms_validated_across_all_outputs=False,
        planned=['single semantic teacher profile','N14 GPU end-to-end parity and matched-access mechanism controls',
                 'crop-only SAM refinement for image queries','matched-geometry independent visibility for old compiled fields']),indent=2)+'\n')
    print(json.dumps(dict(jobs=len(rows),modules=len(modules),output=str(output),model_calls=0)))


if __name__=='__main__':main()
