"""Fixed fine-pixel source ownership control; no new models or field capacity."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from radio_gs.training.fine_pixel_ownership import fine_pixel_ownership
from radio_gs.scripts.prepare_four_teacher_retrieval import WORDS
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.training.source_region_readback import region_render_weights
from radio_gs.utils.immutable_artifacts import sha256_file

OUT=Path('output/fine_pixel_ownership_next32')
RESTORED=Path('output/small_region_restoration_next32')
PILOTS=['lerf_figurines','scannet_scene0000_00']
SOURCES=[__file__,'radio_gs/training/fine_pixel_ownership.py','tests/test_fine_pixel_ownership.py',
 'radio_gs/training/source_pixel_evidence.py','radio_gs/training/source_region_readback.py',
 'radio_gs/scripts/prepare_four_teacher_retrieval.py']


def metric(pred,truth):
    error=pred-truth
    return dict(mse=float(error.square().mean()),centered_mse=float((error-error.mean(1,keepdim=True)).square().mean()),
        teacher_top1_agreement=float((pred.argmax(1)==truth.argmax(1)).float().mean()))


def register():
    OUT.mkdir(exist_ok=False)
    prior=json.loads((RESTORED/'registration.json').read_text())
    reports={r['key']:json.loads((RESTORED/r['key']/'report.json').read_text()) for r in prior['cohort']}
    r=dict(schema='fine_pixel_ownership_next32.registration.v1',source_only=True,pilots=PILOTS,cohort=prior['cohort'],
        inventories='exact original retained inventory plus exact cached restored-small inventory; no new masks or VFM calls',
        relationship='old mask strictly>2x restored-small area, AND intersection>=90% of SMALL area; containment is not identity or negative evidence',
        writing='suppress qualifying old mask only at its intersection with restored small mask; fine masks retained; remaining masks partition each pixel equally',
        invariants='exact pixel coverage union and normalized ownership outside suppressed intersections; per-element source mass conserved relative plain within64FP32eps*max(mass,1)',
        outside_limit='Source writing weights outside affected pixels are exact. Final rendered responses elsewhere need not be exact because the same element can cover multiple pixels.',
        queries=prior['text_queries']+'; '+prior['image_queries'],
        basis_geometry_support='same64basis,geometry,observed support; unobserved rows untouched',
        original_parent_gate=prior['gate'],
        additional_plain_gate='For BOTH modalities and all small/retained/boundary heldout populations: centeredMSE<=plain+1e-6 and teacherTop1>=plain-1e-6. No new pair labels.',
        both_pilots_must_pass=True,no_old_heads_reused=True,benchmark_data_allowed=False,default_promoted=False,
        original_restoration_registration_sha256=sha256_file(RESTORED/'registration.json'),
        parent_seal_sha256=sha256_file('paper/artifacts/small_region_restoration_next32.json'),
        restored={key:dict(small_teacher_sha256=p['small_teacher_sha256'],map_sha256=p['map_sha256'],report_sha256=sha256_file(RESTORED/key/'report.json')) for key,p in reports.items()},
        implementations={p:sha256_file(p) for p in SOURCES})
    (OUT/'registration.json').write_text(json.dumps(r,indent=2)+'\n')


@torch.inference_mode()
def run(row,reg,device):
    key=row['key'];out=OUT/key;out.mkdir(exist_ok=False);root=Path(row['root']);bindings={}
    def load(path,digest):
        assert sha256_file(path)==digest,path;bindings[str(path)]=digest
        return torch.load(path,map_location='cpu',weights_only=False)
    assert sha256_file(root/'teachers.json')==row['teachers_sha256']
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    teacher=load(root/info['teacher_files']['siglip']['path'],info['teacher_files']['siglip']['sha256'])
    assert teacher['queries']==['a photo of a '+word for word in WORDS]
    ep=load(root/'episodes.pt',json.loads((root/'episodes.json').read_text())['sha256'])
    assert ep['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    old_report=json.loads((RESTORED/key/'report.json').read_text())
    assert sha256_file(RESTORED/key/'report.json')==reg['restored'][key]['report_sha256']
    small=load(RESTORED/key/'small_teacher.pt',reg['restored'][key]['small_teacher_sha256'])
    plain_map=load(RESTORED/key/'map.pt',reg['restored'][key]['map_sha256'])
    counts=old_report['source_counts']
    prior=Path('output/native_residual_basis_next30_v2')/key
    mr=json.loads((prior/'map/report.json').read_text());data=load(prior/'map/map.pt',mr['map_sha256'])
    assert data['geometry_binding']==ep['geometry_binding']
    assert plain_map['geometry_binding']==data['geometry_binding']
    assert torch.equal(plain_map['observed'],data['observed']) and torch.equal(plain_map['basis'],data['basis'][:64])
    basis=data['basis'][:64].to(device);features=data['features'][:,:64].to(device);observed=data['observed'].to(device);n=len(observed)
    native=F.normalize(teacher['descriptors'].float().to(device),dim=-1)
    small_native=F.normalize(small['descriptors'].float().to(device),dim=-1)
    compact=native@basis.T;small_compact=small_native@basis.T
    train=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in ep['proposal_views']])
    training=native[train];queries=dict(text=F.normalize(teacher['text_embeddings'].float().to(device),dim=-1),
        image=training[torch.linspace(0,len(training)-1,min(512,len(training))).long()])
    def view(vi):
        r=auth['views'][vi];hit=load(root/r['relative_path'],r['sha256'])
        pid=hit['pixel_ids'].long().to(device);gid=hit['gaussian_ids'].long().to(device);w=hit['base_weights'].float().to(device)
        ids=torch.where(ep['proposal_views']==vi)[0];sid=torch.where(small['proposal_views']==vi)[0]
        masks=ep['masks'][ids].flatten(1).to(device);smasks=small['masks'][sid].flatten(1).to(device)
        mass=torch.zeros(masks.shape[1],device=device).scatter_add_(0,pid,w)
        operator=torch.sparse_coo_tensor(torch.stack([pid,gid]),w/mass[pid].clamp_min(1e-8),(masks.shape[1],n)).coalesce()
        return ids,sid,masks,smasks,pid,gid,w,operator
    numerator={arm:torch.zeros((n,64),device=device) for arm in ('baseline','plain','ownership')}
    masses={arm:torch.zeros(n,device=device) for arm in numerator};fits=[]
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);ids,sid,masks,smasks,pid,gid,w,_=view(vi)
        merged=torch.cat([masks,smasks]);descriptors=torch.cat([compact[ids],small_compact[sid]])
        owned,ownership_record=fine_pixel_ownership(masks,smasks)
        relation=ownership_record.pop('relation');affected=ownership_record.pop('affected_pixels_mask')
        plain_pixel=merged.float()/merged.sum(0).clamp_min(1);owned_pixel=owned.float()/owned.sum(0).clamp_min(1)
        assert torch.equal(plain_pixel[:,~affected],owned_pixel[:,~affected])
        assert torch.allclose(plain_pixel.sum(0),owned_pixel.sum(0),atol=1e-6,rtol=0)
        frame_masses={}
        for arm,mm,cc in [('baseline',masks,compact[ids]),('plain',merged,descriptors),('ownership',owned,descriptors)]:
            lift,_=source_pixel_evidence_weights(mm,pid,gid,w,n,partition_overlap=True)
            numerator[arm]+=lift.T@cc;masses[arm]+=lift.sum(0);frame_masses[arm]=lift.sum(0)
        delta=(frame_masses['plain']-frame_masses['ownership']).abs()
        assert (delta<=64*torch.finfo(torch.float32).eps*frame_masses['plain'].clamp_min(1)).all()
        ownership_record['element_mass_max_error']=float(delta.max())
        fits.append(dict(frame=frame,retained_regions=len(ids),appended_small_regions=len(sid),ownership=ownership_record))
        print(key,'fit',frame,len(sid),flush=True)
    replay=numerator['baseline']/masses['baseline'][:,None].clamp_min(1e-8)
    replay_error=float((replay[observed]-features[observed]).abs().max());assert replay_error<2e-5,replay_error
    assert all((masses[arm][observed]>0).all() for arm in masses)
    plain_replay=numerator['plain']/masses['plain'][:,None].clamp_min(1e-8)
    plain_features=plain_map['features'].to(device)
    plain_replay_error=float((plain_replay[observed]-plain_features[observed]).abs().max());assert plain_replay_error<2e-5,plain_replay_error
    candidate=features.clone();estimate=numerator['ownership']/masses['ownership'][:,None].clamp_min(1e-8)
    candidate[observed]=estimate[observed]
    assert torch.equal(candidate[~observed],features[~observed]) and torch.isfinite(candidate).all()
    torch.save(dict(features=candidate.cpu(),basis=basis.cpu(),observed=data['observed'],geometry_binding=data['geometry_binding'],
        parent_map_sha256=mr['map_sha256'],source_only=True,old_heads_compatible=False),out/'map.pt')
    pools={arm:{kind:{mode:[] for mode in queries} for kind in ('retained','small','boundary')} for arm in ('baseline','plain','ownership')}
    targets={kind:{mode:[] for mode in queries} for kind in ('retained','small','boundary')};observations=[]
    for frame in info['validation_frames']:
        vi=info['frames'].index(frame);ids,sid,masks,smasks,pid,gid,w,operator=view(vi)
        weights,supported,valid=region_render_weights(masks,pid,gid,w,observed)
        if len(sid):sweights,ssupported,svalid=region_render_weights(smasks,pid,gid,w,observed);assert torch.equal(svalid,valid)
        else:sweights=torch.empty((0,n),device=device);ssupported=torch.empty(0,dtype=torch.bool,device=device)
        normalization=torch.sparse.mm(operator,observed.float()[:,None])
        target_vectors=F.normalize((masks.float()/masks.sum(0).clamp_min(1)[None]).T@native[ids],dim=-1)
        h,width=ep['masks'].shape[-2:];bm=masks.float().reshape(-1,1,h,width)
        boundary=((F.max_pool2d(bm,3,1,1)+F.max_pool2d(-bm,3,1,1))[:,0]>0).any(0).flatten()&masks.any(0)&valid
        for mode,q in queries.items():
            targets['retained'][mode].append((native[ids]@q.T)[supported].cpu())
            if len(sid):targets['small'][mode].append((small_native[sid]@q.T)[ssupported].cpu())
            targets['boundary'][mode].append((target_vectors@q.T)[boundary].cpu())
        for arm,values in [('baseline',features),('plain',plain_features),('ownership',candidate)]:
            unit=F.normalize(values,dim=-1);pooled=weights@unit
            small_pooled=sweights@unit
            pixels=torch.sparse.mm(operator,unit*observed[:,None])/normalization.clamp_min(1e-8)
            for mode,q in queries.items():
                projected=q@basis.T
                pools[arm]['retained'][mode].append((pooled@projected.T)[supported].cpu())
                if len(sid):pools[arm]['small'][mode].append((small_pooled@projected.T)[ssupported].cpu())
                pools[arm]['boundary'][mode].append((pixels@projected.T)[boundary].cpu())
        observations.append(dict(frame=frame,retained_regions=len(ids),supported_retained=int(supported.sum()),
            small_regions=len(sid),supported_small=int(ssupported.sum()),unsupported_small_ids=sid[~ssupported.cpu()].tolist(),
            boundary_pixels=int(boundary.sum()),support_identical=True))
        print(key,'heldout',frame,'small',len(sid),int(ssupported.sum()),flush=True)
    assert sum(r['supported_small'] for r in observations)>0
    metrics={arm:{kind:{mode:metric(torch.cat(rows),torch.cat(targets[kind][mode])) for mode,rows in modes.items()}
        for kind,modes in kinds.items()} for arm,kinds in pools.items()}
    gate={}
    for mode in queries:
        gate[mode]={}
        for kind in ('small','retained','boundary'):
            b=metrics['baseline'][kind][mode];c=metrics['ownership'][kind][mode]
            gate[mode][kind]=dict(mse_ratio=c['mse']/max(b['mse'],1e-12),top1_drop=b['teacher_top1_agreement']-c['teacher_top1_agreement'])
    original_gate_pass=all(g['small']['mse_ratio']<=.95 and g['retained']['mse_ratio']<=1.01 and g['boundary']['mse_ratio']<=1.01
        and g['small']['top1_drop']<=.01 and g['retained']['top1_drop']<=.01 for g in gate.values())
    plain_gate={mode:{kind:dict(centered_delta=metrics['ownership'][kind][mode]['centered_mse']-metrics['plain'][kind][mode]['centered_mse'],top1_drop=metrics['plain'][kind][mode]['teacher_top1_agreement']-metrics['ownership'][kind][mode]['teacher_top1_agreement']) for kind in ('small','retained','boundary')} for mode in queries}
    plain_gate_pass=all(g['centered_delta']<=1e-6 and g['top1_drop']<=1e-6 for kinds in plain_gate.values() for g in kinds.values())
    admitted=original_gate_pass and plain_gate_pass
    report=dict(**row,source_only=True,benchmark_data_read=False,source_admitted=admitted,original_gate_pass=original_gate_pass,plain_gate_pass=plain_gate_pass,gate=gate,plain_gate=plain_gate,metrics=metrics,
        observations=observations,source_fits=fits,source_counts=counts,new_small_regions=len(small['masks']),
        training_small_regions=sum(r['appended_small_regions'] for r in fits),validation_small_regions=sum(r['small_regions'] for r in observations),
        support_unchanged=True,basis_unchanged=True,geometry_binding=data['geometry_binding'],original_masks_descriptors_changed=False,
        original_heads_reused=False,map_sha256=sha256_file(out/'map.pt'),small_teacher_sha256=reg['restored'][key]['small_teacher_sha256'],
        baseline_fusion_replay_max_error=replay_error,plain_restoration_replay_max_error=plain_replay_error,registration_sha256=sha256_file(OUT/'registration.json'),inputs=bindings,
        caveat='Teacher pseudo-response preservation, not real small-object localization; original boundary pseudo-target still averages overlapping old regions')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(key,admitted,gate,plain_gate,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--keys',nargs='+',choices=PILOTS)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    if a.register:register();return
    reg=json.loads((OUT/'registration.json').read_text())
    for path,digest in reg['implementations'].items():assert sha256_file(path)==digest,path
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and any(v in str(args[0]) for v in (
            '.labels.ply','/lerf_ovs/label/','/eval_gt/','category_evaluation','ranking_evaluation','/official_annotations/')):
            raise RuntimeError('Benchmark evidence forbidden in small-region source pilot')
    sys.addaudithook(guard)
    assert a.keys
    for row in reg['cohort']:
        if row['key'] in a.keys:run(row,reg,a.device)


if __name__=='__main__':main()
