"""Preregistered source-only absolute-opacity evidence fusion pilot.

No benchmark reads, independent adapter, query-specific map or threshold sweep.
The single objective is fixed before source results; step zero remains eligible.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from radio_gs.training.alpha_weighted_source_evidence import alpha_weighted_source_evidence
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.training.source_region_readback import region_render_weights
from radio_gs.utils.immutable_artifacts import sha256_file


OUTPUT = Path('output/alpha_weighted_source_fusion_next32')
IMPLEMENTATIONS = [__file__, 'radio_gs/training/alpha_weighted_source_evidence.py',
                  'radio_gs/training/source_pixel_evidence.py',
                  'tests/test_alpha_weighted_source_evidence_next32.py',
                  'radio_gs/training/source_region_readback.py']
PILOTS = ['lerf_figurines','scannet_scene0000_00']


def evaluate(pred,truth):
    error=pred-truth
    return dict(mse=float(error.square().mean()),
        centered_mse=float((error-error.mean(1,keepdim=True)).square().mean()),
        teacher_top1_agreement=float((pred.argmax(1)==truth.argmax(1)).float().mean()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--keys',nargs='+')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.register:
        OUTPUT.mkdir(exist_ok=False)
        cohort=json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())['cohort']
        registration=dict(schema='source_alpha_weighted_fusion.next32',source_only=True,pilots=PILOTS,
            cohort=[r for r in cohort if r['benchmark'] in ('lerf','scannet')],
            objective='Replace source writing weight T*alpha/A with raw T*alpha; retain mask overlap partition and original weak element predicate. No parameters or sweep.',
            deployment='Identical observed-normalized render and native cosine query norm. Changed element vectors require dependent head refit before target use.',
            capacity='unchanged64 native dimensions, basis and support',
            source_selection='Each pilot independently needs >=1% relative heldout-region text and image MSE improvement; centered MSE nonincrease; teacher top1 decrease <=1pp; covered and boundary pixel text/image MSE nonincrease and top1 drop <=1pp; unchanged observed support. No benchmark-based switch. Otherwise retain step zero.',
            expansion='Full12 only if BOTH registered pilots pass their source gate',
            pixel_boundary_metrics='Source pseudo regional teacher responses and SAM boundaries are not pixel semantics or physical truth; they are an additional conservative source gate and do not establish selected-object IoU.',
            alpha_diagnostics='Covered-pixel alpha quantiles; alpha<.1/.01 normalized vs raw vote mass share. Thresholds diagnostic only.',
            prior_control_distinction='Prior contribution/partition controls kept T*alpha/A. Historical raw-contribution diagnostic used different crops and basis, not this matched estimator.',
            implementations={str(p):sha256_file(p) for p in IMPLEMENTATIONS})
        (OUTPUT/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
        print('registered',PILOTS,flush=True)
        return
    registration=json.loads((OUTPUT/'registration.json').read_text())
    for path,digest in registration['implementations'].items():
        assert sha256_file(path)==digest,path
    assert args.keys
    if any(key not in PILOTS for key in args.keys):
        assert all(json.loads((OUTPUT/key/'report.json').read_text())['source_admitted'] for key in PILOTS)
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)) and any(x in str(values[0]) for x in (
                '.labels.ply','/lerf_ovs/label/','category_evaluation','ranking_evaluation')):
            raise RuntimeError('Target evidence forbidden in source fusion candidate')
    sys.addaudithook(guard)
    for row in registration['cohort']:
        if row['key'] not in args.keys:
            continue
        with torch.inference_mode():
            run(row,registration,args.device)


def run(row,registration,device):
    key=row['key'];out=OUTPUT/key;out.mkdir(exist_ok=False)
    root=Path(row['root'])
    info=json.loads((root/'teachers.json').read_text())
    auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    bindings={}
    def load(path,digest):
        assert sha256_file(path)==digest,path
        bindings[str(path)]=digest
        return torch.load(path,map_location='cpu',weights_only=False)
    teacher=load(root/info['teacher_files']['siglip']['path'],info['teacher_files']['siglip']['sha256'])
    ep=load(root/'episodes.pt',json.loads((root/'episodes.json').read_text())['sha256'])
    assert ep['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    previous=Path('output/native_residual_basis_next30_v2')/key
    mr=json.loads((previous/'map/report.json').read_text())
    data=load(previous/'map/map.pt',mr['map_sha256'])
    assert data['geometry_binding']==ep['geometry_binding']
    basis=data['basis'][:64].to(device)
    observed=data['observed'].to(device)
    features=data['features'][:,:64].to(device)
    n=len(observed)
    native=F.normalize(teacher['descriptors'].float().to(device),dim=-1)
    compact=native @ basis.T
    train=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in ep['proposal_views']])
    training=native[train]
    queries=dict(text=F.normalize(teacher['text_embeddings'].float().to(device),dim=-1),
        image=training[torch.linspace(0,len(training)-1,min(512,len(training))).long()])
    numerator=torch.zeros((n,64),device=device)
    baseline_numerator=torch.zeros_like(numerator)
    mass=torch.zeros(n,device=device)
    candidate_mass=torch.zeros_like(mass)
    fits=[]
    def view(vi):
        rec=auth['views'][vi]
        hit=load(root/rec['relative_path'],rec['sha256'])
        pid=hit['pixel_ids'].long().to(device);gid=hit['gaussian_ids'].long().to(device)
        contribution=hit['base_weights'].float().to(device)
        ids=torch.where(ep['proposal_views']==vi)[0]
        masks=ep['masks'][ids].flatten(1).to(device)
        pixels=masks.shape[1]
        pixelmass=torch.zeros(pixels,device=device).scatter_add_(0,pid,contribution)
        operator=torch.sparse_coo_tensor(torch.stack([pid,gid]),contribution/pixelmass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
        return ids,masks,pid,gid,contribution,operator
    for frame in info['train_frames']:
        vi=info['frames'].index(frame)
        ids,masks,pid,gid,contribution,operator=view(vi)
        baseline_weights,_=source_pixel_evidence_weights(masks,pid,gid,contribution,n,partition_overlap=True)
        candidate_weights,pixel_mass=alpha_weighted_source_evidence(masks,pid,gid,contribution,n)
        baseline_numerator+=baseline_weights.T @ compact[ids]
        numerator+=candidate_weights.T @ compact[ids]
        mass+=baseline_weights.sum(0)
        candidate_mass+=candidate_weights.sum(0)
        covered=masks.any(0)
        seen=torch.sparse.sum(operator,dim=0).to_dense()>1e-6
        eligible=covered[pid]&seen[gid]&observed[gid]
        normvotes=contribution/pixel_mass[pid].clamp_min(1e-8)
        q=torch.tensor([0.,.01,.1,.5,.9,.99,1.],device=device)
        fit=dict(frame=frame,covered_pixels=int(covered.sum()),
            covered_zero_alpha_pixels=int((covered&(pixel_mass<=1e-8)).sum()),
            covered_alpha_quantiles=dict(zip(['min','p01','p10','p50','p90','p99','max'],pixel_mass[covered].quantile(q).tolist())),
            baseline_vote_mass=float(normvotes[eligible].sum()),candidate_vote_mass=float(contribution[eligible].sum()),
            low_alpha={str(threshold):dict(pixels=int((covered&(pixel_mass<threshold)).sum()),
                baseline_vote_mass=float(normvotes[eligible&(pixel_mass[pid]<threshold)].sum()),
                candidate_vote_mass=float(contribution[eligible&(pixel_mass[pid]<threshold)].sum())) for threshold in (.1,.01)})
        fits.append(fit)
        print(key,'source frame',frame,flush=True)
    estimate=baseline_numerator/mass[:,None].clamp_min(1e-8)
    replay_error=float((estimate[observed]-features[observed]).abs().max())
    assert replay_error<2e-5,('Baseline fusion does not reproduce',replay_error)
    assert (mass[observed]>0).all()
    candidate=features.clone()
    assert (candidate_mass[observed]>0).all()
    candidate[observed]=(numerator/candidate_mass[:,None].clamp_min(1e-8))[observed]
    assert torch.isfinite(candidate).all()
    torch.save(dict(features=candidate.cpu(),basis=basis.cpu(),observed=data['observed'],
        geometry_binding=data['geometry_binding'],parent_map_sha256=mr['map_sha256'],
        source_only=True,original_heads_compatible=False),out/'map.pt')
    pools={arm:{mode:[] for mode in queries} for arm in ('baseline','candidate')}
    truths={mode:[] for mode in queries}
    pixel_rows=[];observations=[]
    for frame in info['validation_frames']:
        vi=info['frames'].index(frame)
        ids,masks,pid,gid,contribution,operator=view(vi)
        weights,supported,valid=region_render_weights(masks,pid,gid,contribution,observed)
        normalization=torch.sparse.mm(operator,observed.float()[:,None])
        pixel_target=F.normalize((masks.float()/masks.sum(0).clamp_min(1)[None]).T @ native[ids],dim=-1)
        h,w=ep['masks'].shape[-2:]
        binary=masks.float().reshape(-1,1,h,w)
        boundary=((F.max_pool2d(binary,3,1,1)+F.max_pool2d(-binary,3,1,1))[:,0]>0).any(0).flatten()
        covered=masks.any(0)&valid
        strata=dict(covered=covered,boundary=covered&boundary,interior=covered&~boundary)
        for mode,q in queries.items():
            truths[mode].append((native[ids] @ q.T)[supported].cpu())
        for arm,field in [('baseline',features),('candidate',candidate)]:
            unit=F.normalize(field,dim=-1)
            pooled=weights @ unit
            rendered=torch.sparse.mm(operator,unit*observed[:,None])/normalization.clamp_min(1e-8)
            for mode,q in queries.items():
                projected=q @ basis.T
                pools[arm][mode].append((pooled @ projected.T)[supported].cpu())
                pixel_prediction=rendered @ projected.T
                target=pixel_target @ q.T
                for stratum,subset in strata.items():
                    if subset.any():
                        pixel_rows.append(dict(frame=frame,arm=arm,modality=mode,stratum=stratum,
                            pixels=int(subset.sum()),**evaluate(pixel_prediction[subset],target[subset])))
        observations.append(dict(frame=frame,regions=len(ids),supported_regions=int(supported.sum()),
            unsupported_region_ids=ids[~supported.cpu()].tolist(),covered_pixels=int(covered.sum()),
            boundary_pixels=int(strata['boundary'].sum()),observed_pixel_support_exact=True))
    metrics={arm:{mode:evaluate(torch.cat(values),torch.cat(truths[mode]))
        for mode,values in modes.items()} for arm,modes in pools.items()}
    gate={mode:dict(mse_ratio=metrics['candidate'][mode]['mse']/max(metrics['baseline'][mode]['mse'],1e-12),
        centered_nonincrease=metrics['candidate'][mode]['centered_mse']<=metrics['baseline'][mode]['centered_mse'],
        top1_drop=metrics['baseline'][mode]['teacher_top1_agreement']-metrics['candidate'][mode]['teacher_top1_agreement'])
        for mode in queries}
    pixel_metrics={arm:{mode:{} for mode in queries} for arm in ('baseline','candidate')}
    for arm in pixel_metrics:
        for mode in queries:
            for stratum in ('covered','boundary','interior'):
                rows=[r for r in pixel_rows if r['arm']==arm and r['modality']==mode and r['stratum']==stratum]
                total=sum(r['pixels'] for r in rows)
                pixel_metrics[arm][mode][stratum]=dict(pixels=total,**{metric:sum(r['pixels']*r[metric] for r in rows)/max(total,1) for metric in ('mse','centered_mse','teacher_top1_agreement')})
    pixel_gate={mode:{stratum:dict(
        mse_ratio=pixel_metrics['candidate'][mode][stratum]['mse']/max(pixel_metrics['baseline'][mode][stratum]['mse'],1e-12),
        top1_drop=pixel_metrics['baseline'][mode][stratum]['teacher_top1_agreement']-pixel_metrics['candidate'][mode][stratum]['teacher_top1_agreement'])
        for stratum in ('covered','boundary')} for mode in queries}
    admitted=all(g['mse_ratio']<=.99 and g['centered_nonincrease'] and g['top1_drop']<=.01 for g in gate.values()) and all(g['mse_ratio']<=1 and g['top1_drop']<=.01 for mode in pixel_gate.values() for g in mode.values())
    alpha_totals=dict(baseline_vote_mass=sum(r['baseline_vote_mass'] for r in fits),candidate_vote_mass=sum(r['candidate_vote_mass'] for r in fits),covered_pixels=sum(r['covered_pixels'] for r in fits))
    alpha_totals['low_alpha']={str(threshold):dict(
        pixel_fraction=sum(r['low_alpha'][str(threshold)]['pixels'] for r in fits)/max(alpha_totals['covered_pixels'],1),
        **{arm+'_vote_fraction':sum(r['low_alpha'][str(threshold)][arm+'_vote_mass'] for r in fits)/max(alpha_totals[arm+'_vote_mass'],1e-12) for arm in ('baseline','candidate')}) for threshold in (.1,.01)}
    report=dict(**row,source_only=True,target_evidence_used=False,query_independent_map=True,
        source_admitted=admitted,chosen_step=1 if admitted else 0,gate=gate,metrics=metrics,
        source_fits=fits,alpha_diagnostics=alpha_totals,pixel_metrics=pixel_metrics,pixel_gate=pixel_gate,
        pixel_boundary_diagnostics=pixel_rows,observations=observations,
        baseline_fusion_replay_max_error=replay_error,support_exact=True,basis_exact=True,
        maximum_element_change=float((candidate-features).norm(dim=1).max()),
        maximum_candidate_element_norm=float(candidate.norm(dim=1).max()),
        no_coordinate_clipping=True,original_heads_compatible=False,
        raw_mass_min_observed=float(candidate_mass[observed].min()),
        map_sha256=sha256_file(out/'map.pt'),registration_sha256=sha256_file(OUTPUT/'registration.json'),
        geometry_binding=data['geometry_binding'],inputs=bindings,
        caveat='Source pseudo responses, no target or final selected-object evaluation. Passing source gate only permits dependent-head refit and further evaluation.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(key,'admitted',admitted,gate,flush=True)


if __name__=='__main__':
    main()
