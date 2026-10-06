"""Two fixed source pilots for a query-independent two-mode semantic field."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from radio_gs.training.native_semantic_mixture import fit_weighted_native_two_means, native_mixture_scores
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.training.source_region_readback import region_render_weights
from radio_gs.utils.immutable_artifacts import sha256_file

OUT=Path('output/native_mixture_next31')
PILOTS=['lerf_figurines','scannet_scene0000_00']
SOURCES=[__file__,'radio_gs/training/native_semantic_mixture.py',
         'radio_gs/training/source_pixel_evidence.py','radio_gs/training/source_region_readback.py']


def metrics(pred,truth):
    error=pred-truth
    return dict(mse=float(error.square().mean()),
        centered_mse=float((error-error.mean(1,keepdim=True)).square().mean()),
        over_response_mse=float(error.clamp_min(0).square().mean()),
        teacher_top1_agreement=float((pred.argmax(1)==truth.argmax(1)).float().mean()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true')
    p.add_argument('--keys',nargs='+',choices=PILOTS);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(4)
    if a.register:
        OUT.mkdir(exist_ok=False)
        cohort=json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())['cohort']
        registration=dict(schema='native_semantic_mixture.next31.v1',pilots=PILOTS,
            cohort=[r for r in cohort if r['key'] in PILOTS],source_only=True,
            method='partition-overlap renderer source masses; spherical weighted2means with raw projected-coordinate centroids;8Lloyd updates; initialize weighted mean and max(weight*(1-cosine_to_mean)) source descriptor, smallestregionid tie break',
            temperature='global weighted mean training assigned-mode cosine deficit1-cos;floor1e-4 only;nothing fitted on validation or benchmark;energy-scale heuristic,not calibrated posterior',
            scores='tau*log(sum_k prior_k*exp(native_component_cosine_k/tau));nativequery normalized beforeprojection;scoreelements before renderer',
            capacity='160 float32 coordinates per element (2x64semantic+32copied structural) plus1 float32mode prior:644bytes vs384bytes parent;global64x1536basis unchanged',
            support='exactly parent observed;unseen source rows never admitted',
            source_gate='BOTH pilots independently: text ANDimage regionMSE<=.99baseline,centeredMSE nonincrease,teacherTop1drop<=.01;eachmodality boundaryMSE and boundaryover-responseMSE nonincrease;unchanged support',
            over_response='max(predictedcosine-teacherpseudoresponse,0)^2;not labeled objectFPR',
            image_queries='512 evenlyspaced training regionalnative descriptors;fixed source textbank',
            benchmark_evaluation_permitted_only_if_both_source_pilots_pass=True,
            original_calibration_scope_heads_reused=False,default_promoted=False,
            implementations={str(p):sha256_file(p) for p in SOURCES})
        (OUT/'registration.json').write_text(json.dumps(registration,indent=2)+'\n');return
    reg=json.loads((OUT/'registration.json').read_text())
    for path,digest in reg['implementations'].items():assert sha256_file(path)==digest,path
    assert a.keys
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in (
            '.labels.ply','/lerf_ovs/label/','category_evaluation','ranking_evaluation','/eval_gt/')):
            raise RuntimeError('Benchmark evidence forbidden in source mixture fit')
    sys.addaudithook(guard)
    for row in reg['cohort']:
        if row['key'] in a.keys:
            with torch.inference_mode():run(row,reg,a.device)


def run(row,reg,device):
    key=row['key'];out=OUT/key;out.mkdir(exist_ok=False);root=Path(row['root']);bindings={}
    def load(path,digest):
        assert sha256_file(path)==digest,path;bindings[str(path)]=digest
        return torch.load(path,map_location='cpu',weights_only=False)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    teacher=load(root/info['teacher_files']['siglip']['path'],info['teacher_files']['siglip']['sha256'])
    ep=load(root/'episodes.pt',json.loads((root/'episodes.json').read_text())['sha256'])
    assert ep['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    previous=Path('output/native_residual_basis_next30_v2')/key;mr=json.loads((previous/'map/report.json').read_text())
    data=load(previous/'map/map.pt',mr['map_sha256']);assert data['geometry_binding']==ep['geometry_binding']
    basis=data['basis'][:64].to(device);features=data['features'][:,:64].to(device);observed=data['observed'].to(device);n=len(observed)
    native=F.normalize(teacher['descriptors'].float().to(device),dim=-1);compact=native@basis.T
    train=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in ep['proposal_views']])
    training=native[train];queries=dict(text=F.normalize(teacher['text_embeddings'].float().to(device),dim=-1),
        image=training[torch.linspace(0,len(training)-1,min(512,len(training))).long()])
    def view(vi):
        r=auth['views'][vi];hit=load(root/r['relative_path'],r['sha256'])
        pid=hit['pixel_ids'].long().to(device);gid=hit['gaussian_ids'].long().to(device);w=hit['base_weights'].float().to(device)
        ids=torch.where(ep['proposal_views']==vi)[0];masks=ep['masks'][ids].flatten(1).to(device)
        full=torch.zeros(masks.shape[1],device=device).scatter_add_(0,pid,w)
        operator=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/full[pid].clamp_min(1e-8),(masks.shape[1],n)).coalesce()
        return ids,masks,pid,gid,w,operator
    ei=[];ri=[];ww=[];frame_votes=[]
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);ids,masks,pid,gid,w,_=view(vi)
        lift,_=source_pixel_evidence_weights(masks,pid,gid,w,n,partition_overlap=True)
        lift[:,~observed]=0
        region,element=torch.where(lift>0);weight=lift[region,element]
        ei.append(element.cpu());ri.append(ids[region.cpu()]);ww.append(weight.cpu())
        frame_votes.append(dict(frame=frame,votes=len(weight),source_mass=float(weight.double().sum())))
        print(key,'votes',frame,len(weight),flush=True)
        del lift
    ei=torch.cat(ei).to(device);ri=torch.cat(ri).to(device);ww=torch.cat(ww).to(device)
    assert train[ri.cpu()].all(), 'Validation source evidence entered mode fit'
    fit=fit_weighted_native_two_means(compact,ei,ri,ww,n,iterations=8)
    centers=fit['centers'];priors=fit['mixture_weights'];tau=fit['temperature']
    replay=float((fit['mean'][observed]-features[observed]).abs().max())
    assert replay<2e-5,('Exact same source mean failed',replay)
    assert (fit['mass'][observed]>0).all()
    del ei,ri,ww
    # Structural coordinates are copied only for explicit storage accounting;
    # no prior structural query/calibration head is asserted compatible.
    fp=root/row['run']/'field.pt';fr=json.loads(fp.with_name('report.json').read_text());parent=load(fp,fr['field_sha256'])
    latent=parent['state_dict']['latent'];assert latent.shape==(n,96)
    vector=torch.cat([centers.cpu().reshape(n,128),latent[:,64:96].float()],dim=1)
    torch.save(dict(latent=vector,mode_one_prior=priors[:,1].cpu(),basis=basis.cpu(),temperature=tau,
        observed=data['observed'],geometry_binding=data['geometry_binding'],source_only=True,
        old_query_heads_compatible=False,structural_suffix_copied_not_revalidated=True),out/'map.pt')
    del parent,vector
    region_predictions={a:{m:[] for m in queries} for a in ['baseline','mixture']};truths={m:[] for m in queries}
    pixel_predictions={a:{m:{s:[] for s in ['covered','boundary','interior']} for m in queries} for a in ['baseline','mixture']}
    pixel_truths={m:{s:[] for s in ['covered','boundary','interior']} for m in queries};observations=[]
    unit=F.normalize(features,dim=-1)
    for frame in info['validation_frames']:
        vi=info['frames'].index(frame);ids,masks,pid,gid,w,operator=view(vi)
        weights,supported,valid=region_render_weights(masks,pid,gid,w,observed)
        normalization=torch.sparse.mm(operator,observed.float()[:,None])
        target_vectors=F.normalize((masks.float()/masks.sum(0).clamp_min(1)[None]).T@native[ids],dim=-1)
        h,width=ep['masks'].shape[-2:];binary=masks.float().reshape(-1,1,h,width)
        boundary=((F.max_pool2d(binary,3,1,1)+F.max_pool2d(-binary,3,1,1))[:,0]>0).any(0).flatten()
        covered=masks.any(0)&valid;strata=dict(covered=covered,boundary=covered&boundary,interior=covered&~boundary)
        base_region=weights@unit;base_pixel=torch.sparse.mm(operator,unit*observed[:,None])/normalization.clamp_min(1e-8)
        for modality,q in queries.items():
            truth=native[ids]@q.T;truths[modality].append(truth[supported].cpu())
            pixel_target=(target_vectors@q.T).cpu();projected=q@basis.T
            region_predictions['baseline'][modality].append((base_region@projected.T)[supported].cpu())
            baseline_pixel=(base_pixel@projected.T).cpu();mixed_region=[];mixed_pixel=[]
            for start in range(0,len(q),64):
                values=native_mixture_scores(centers,priors,q[start:start+64],basis,tau)
                mixed_region.append((weights@values)[supported].cpu())
                mixed_pixel.append((torch.sparse.mm(operator,values*observed[:,None])/normalization.clamp_min(1e-8)).cpu())
            region_predictions['mixture'][modality].append(torch.cat(mixed_region,dim=1));mixture_pixel=torch.cat(mixed_pixel,dim=1)
            for name,mask in strata.items():
                cpu=mask.cpu()
                if not cpu.any():continue
                pixel_truths[modality][name].append(pixel_target[cpu])
                pixel_predictions['baseline'][modality][name].append(baseline_pixel[cpu])
                pixel_predictions['mixture'][modality][name].append(mixture_pixel[cpu])
        observations.append(dict(frame=frame,regions=len(ids),supported_regions=int(supported.sum()),
            unsupported_region_ids=ids[~supported.cpu()].tolist(),covered_pixels=int(covered.sum()),boundary_pixels=int(strata['boundary'].sum())))
        print(key,'heldout',frame,flush=True)
    summary={a:{m:metrics(torch.cat(v),torch.cat(truths[m])) for m,v in modalities.items()} for a,modalities in region_predictions.items()}
    pixels={a:{m:{s:metrics(torch.cat(v),torch.cat(pixel_truths[m][s])) for s,v in strata.items() if v} for m,strata in modes.items()} for a,modes in pixel_predictions.items()}
    gate={}
    for modality in queries:
        b=summary['baseline'][modality];c=summary['mixture'][modality];bp=pixels['baseline'][modality]['boundary'];cp=pixels['mixture'][modality]['boundary']
        gate[modality]=dict(region_mse_ratio=c['mse']/max(b['mse'],1e-12),
            region_centered_nonincrease=c['centered_mse']<=b['centered_mse'],teacher_top1_drop=b['teacher_top1_agreement']-c['teacher_top1_agreement'],
            boundary_mse_nonincrease=cp['mse']<=bp['mse'],boundary_over_response_nonincrease=cp['over_response_mse']<=bp['over_response_mse'])
    admitted=all(g['region_mse_ratio']<=.99 and g['region_centered_nonincrease'] and g['teacher_top1_drop']<=.01 and g['boundary_mse_nonincrease'] and g['boundary_over_response_nonincrease'] for g in gate.values())
    gap=1-(F.normalize(centers[:,0],dim=-1)*F.normalize(centers[:,1],dim=-1)).sum(-1)
    minimum_prior=priors.min(-1).values
    report=dict(**row,source_only=True,benchmark_data_read=False,source_admitted=admitted,
        metrics=summary,pixel_metrics=pixels,gate=gate,observations=observations,training_frames=info['train_frames'],validation_frames=info['validation_frames'],source_votes=frame_votes,
        temperature=tau,temperature_unfloored=fit['temperature_unfloored'],iterations=8,
        two_nonempty_modes=fit['two_nonempty_modes'],observed_elements=int(observed.sum()),
        bimodal_diagnostic_fraction=float(((minimum_prior>.1)&(gap>.1))[observed].float().mean()),
        bimodal_diagnostic_definition='both source mass priors>.1 and intermode cosine<.9;report only,never admission or scoring',
        minimum_prior_quantiles=torch.quantile(minimum_prior[observed],torch.tensor([0.,.25,.5,.75,1.],device=device)).cpu().tolist(),
        intermode_cosine_deficit_quantiles=torch.quantile(gap[observed],torch.tensor([0.,.25,.5,.75,1.],device=device)).cpu().tolist(),
        old_source_mean_replay_max_error=replay,weighted_mode_mean_reconstruction_max_error=fit['weighted_mean_reconstruction_max_error'],
        support_unchanged=True,basis_unchanged=True,parent_coordinates_per_element=96,candidate_coordinates_per_element=160,
        extra_prior_scalars_per_element=1,parent_float32_bytes_per_element=384,candidate_float32_bytes_per_element=644,
        candidate_float32_element_bytes=n*644,parent_float32_element_bytes=n*384,original_heads_reused=False,
        map_sha256=sha256_file(out/'map.pt'),registration_sha256=sha256_file(OUT/'registration.json'),inputs=bindings,
        caveat='Pseudo-response development preflight;not calibrated probabilities,real semantic FPR,benchmark gain,or finalfour-query field')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(key,'admitted',admitted,gate,flush=True)


if __name__=='__main__':main()
