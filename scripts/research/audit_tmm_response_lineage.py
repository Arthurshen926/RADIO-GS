"""Audit where retained provider responses already consume refinement evidence.

This is input-lineage evidence, not a causal estimate of accuracy loss. It does
not claim that all reuse is invalid or statistically independent evidence exists.
"""
from pathlib import Path
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from radio_gs.querying.response_study_adapters import SYSTEM,read
from scripts.research.replay_tmm_response import STUDY,POLICY,sha


def main():
    rows=[];hashes={}
    for job in read('configs/pipeline/system_v10.json')['jobs']:
        family=job['benchmark'];base=SYSTEM/family/job['name']/'query'
        hashes[job['scene']]=sha(job['scene'])
        scene=read(job['scene'])
        row=dict(family=family,scene=job['name'],new_teacher_calls_in_post_study=False)
        if family=='lerf':
            path=base/'text/response/fit.json';fit=read(path)
            old=read(fit['inputs']);new=read(base/'text/identity_evidence/inputs.json')
            assert Path(old['detections']).resolve()==Path(new['detections']).resolve()
            assert fit['source_frames']==read(new['scene'])['source_frames']
            row.update(already_consumed='same identity-admitted masks and source frames',
                producer='radio_gs.pipeline.response',source_views=len(fit['source_frames']),
                original_response_input=fit['readout'],original_response_input_exists=Path(fit['readout']).exists(),
                intervention_site='replace response fusion after fixed identity admission; retain independent sets')
        elif family=='nvos':
            path=base/'reference/0/fit.json';fit=read(path)
            assert fit['source_frames']==scene['source_frames']
            row.update(already_consumed='reference/track mask estimates used by rendered response solver',
                producer='radio_gs.querying.reference_response',source_views=len(fit['source_frames']),
                original_response_input='neutral 0.5 response on frozen geometry before rendered evidence solve',
                intervention_site='matched evidence solve; preserve reference and track inputs plus independent sets')
        elif family=='scannet':
            path=base/'category/result.json';report=read(path)
            assert report['source_solver']['method']=='equal_native_prior_control'
            assert report['source_frames']==[v['frame'] for v in scene['source_evidence']['views']]
            assert report['source_solver']['local_auxiliary_scene']==scene['local_auxiliary_scene']
            row.update(already_consumed='same query-conditioned source mask moments in compile_local_evidence',
                producer='radio_gs.querying.local_element_fusion',source_views=len(report['source_frames']),
                original_response_input='both native priors BEFORE source fusion; do not substitute saved single-native logits',
                intervention_site='retain both native teachers; compare source-admission/fusion at matched inputs')
        else:
            path=base/'registered/runs'/f"{job['name']}.json";report=read(path)
            assert scene.get('direct_anchor_membership') and 'image' in scene.get('inferred_anchor_membership',[])
            details=report['query_details']
            row.update(already_consumed='same map membership/visibility relations in complete-anchor membership features; text matched region scope',
                producer='DirectAnchorEvidence.encode / joint_anchor_evidence; InstanceObservationScope.decode',
                query_count=len(details),
                original_response_input='frozen native identity AND complete-membership provider; shared evidence is not a new teacher',
                intervention_site='test complementary query-response information or calibrated conditional correction; keep full-member capability',
                prohibited_shortcut='replacing complete-member response with raw native similarity to manufacture headroom')
        hashes[str(path)]=sha(path);row['provenance_record']=str(path);rows.append(row)
    sources=['radio_gs/pipeline/response.py','radio_gs/querying/reference_response.py',
        'radio_gs/querying/local_element_fusion.py','radio_gs/querying/direct_anchor_membership.py',
        'radio_gs/querying/anchor_observation_evidence.py','radio_gs/querying/instance_observation_scope.py',
        'scripts/research/audit_tmm_response_lineage.py']
    hashes.update({p:sha(p) for p in sources})
    run=BoundedExperiment(STUDY,'response_lineage_audit',POLICY)
    result=dict(jobs=len(rows),rows=rows,source_hashes=hashes,ground_truth_read=False,
        conclusion='all post-refinement families reuse evidence already consumed upstream',
        loss_causality_proven=False,repeated_evidence_is_not_always_harmful=True,
        preserve_strong_provider_baseline=True)
    run.json('summary.json',result)
    print('Verified response evidence lineage for',len(rows),'jobs')


if __name__=='__main__':main()
