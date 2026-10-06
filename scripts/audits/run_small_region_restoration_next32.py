"""Two source-only, fixed-basis pilots restoring pre-encoding small SAM masks."""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio_gs.training.small_region_inventory import discarded_small_region_ids,original_native_gray_crop
from radio_gs.training.source_sam_evidence import read_manifest,load_source_sam
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.scripts.prepare_four_teacher_retrieval import WORDS
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.training.source_region_readback import region_render_weights
from radio_gs.utils.mask_resampling import resize_boolean_masks_nearest_exact
from radio_gs.utils.immutable_artifacts import sha256_file

OUT=Path('output/small_region_restoration_next32')
PILOTS=['lerf_figurines','scannet_scene0000_00']
SOURCES=[__file__,'radio_gs/training/small_region_inventory.py','tests/test_small_region_inventory.py',
 'radio_gs/training/source_sam_evidence.py','radio_gs/training/source_pixel_evidence.py',
 'radio_gs/training/source_region_readback.py','radio_gs/utils/mask_resampling.py',
 'radio_gs/scripts/build_multiscale_sam_mask_aligned_crop_summary_teacher.py',
 'radio_gs/scripts/prepare_four_teacher_retrieval.py','radio_gs/scripts/build_sam3_automatic_mask_cache.py']


def metric(pred,truth):
    error=pred-truth
    return dict(mse=float(error.square().mean()),centered_mse=float((error-error.mean(1,keepdim=True)).square().mean()),
        teacher_top1_agreement=float((pred.argmax(1)==truth.argmax(1)).float().mean()))


def register():
    OUT.mkdir(exist_ok=False)
    cohort=json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())['cohort']
    rows=[]
    for row in cohort:
        if row['key'] not in PILOTS:continue
        parent=Path(row['root']).parents[1]
        raw=parent/'base/raw_field';sam_manifest=parent/'base/source_sam/manifest.json'
        info=json.loads((raw/'teachers.json').read_text())
        assert sha256_file(sam_manifest)==info['sam_source_manifest_sha256']
        rows.append(dict(row,raw_root=str(raw),sam_manifest=str(sam_manifest),
            sam_manifest_sha256=sha256_file(sam_manifest),teachers_sha256=sha256_file(Path(row['root'])/'teachers.json'),
            raw_teachers_sha256=sha256_file(raw/'teachers.json'),source_frames_sha256=sha256_file(Path(row['root'])/'source_frames.json')))
    r=dict(schema='small_region_restoration_next32.registration.v1',source_only=True,pilots=PILOTS,cohort=rows,
        intervention='Append query-free native SAM proposals with nonempty nearest-exact96x128 mask area<.002; existing masks/descriptors unchanged; retain >.6 exclusion',
        encoding='same native gray127 bbox±3 crop,384BICUBIC,original SigLIP2 vision bundle,half descriptor storage,then native normalization and original64projection',
        map='same source normalized pixel-partition weights,train views only,exact observed domain/geometry/basis; no opacity change or recalibration',
        text_queries='existing fixed WORDS bank; assert exact matching text strings',
        image_queries='512 evenly spaced ORIGINAL retained training native descriptors; exclude new and heldout descriptors',
        evaluation='same heldout small masks/native descriptors for both maps; original retained region and boundary targets unchanged; actual normalized renderer then region averaging',
        gate='BOTH pilots independently, text AND image: small-regionMSE<=.95parent; original-retained-regionMSE and original-boundaryMSE<=1.01parent; small and retainedTop1drop<=.01; exact support/basis; no unsupported-region dropping difference',
        gate_small_regions_need_nonzero_evaluation_population=True,
        pseudo_labels_are_not_ground_truth=True,benchmark_data_allowed=False,source_heads_reused=False,
        independent_test=False,default_promoted=False,full_refit_authorized_only_if_both_source_pilots_pass=True,
        implementations={p:sha256_file(p) for p in SOURCES})
    (OUT/'registration.json').write_text(json.dumps(r,indent=2)+'\n')


def recover_small(row,info,teacher,bindings,device,out):
    root=Path(row['root']);raw=Path(row['raw_root'])
    assert sha256_file(raw/'teachers.json')==row['raw_teachers_sha256']
    raw_info=json.loads((raw/'teachers.json').read_text())
    assert raw_info['frames']==info['frames'] and raw_info['train_frames']==info['train_frames'] and raw_info['validation_frames']==info['validation_frames']
    assert sha256_file(row['sam_manifest'])==row['sam_manifest_sha256']==info['sam_source_manifest_sha256']
    document,records=read_manifest(row['sam_manifest'])
    source=json.loads((root/'source_frames.json').read_text())
    assert sha256_file(root/'source_frames.json')==row['source_frames_sha256']==info['source_manifest_sha256']
    source_by_frame={r['frame_idx']:r for r in source['frames']}
    authority=json.loads((root/'responsibility.json').read_text());h=authority['metadata']['feature_height'];w=authority['metadata']['feature_width']
    assert (h,w)==(96,128)
    raw_sam_path=raw/raw_info['teacher_files']['sam']['path'];assert sha256_file(raw_sam_path)==raw_info['teacher_files']['sam']['sha256']
    raw_sam=torch.load(raw_sam_path,map_location='cpu',weights_only=False)
    masks=[];views=[];crops=[];rows=[];counts=[]
    for vi,frame in enumerate(info['frames']):
        src=source_by_frame[frame];image_path=Path(source['image_dir'])/src['source_file']
        sam,sam_hash=load_source_sam(document,records,image_path,src['source_sha256'])
        bindings[str(image_path)]=src['source_sha256'];bindings[records[str(image_path.resolve())]['output']]=sam_hash
        image=Image.open(image_path).convert('RGB');native=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])
        assert tuple(native.shape[1:])==(image.height,image.width)
        resized=resize_boolean_masks_nearest_exact(torch.from_numpy(native.copy()),(h,w))
        original=torch.where(raw_sam['view_indices']==vi)[0]
        old_native_ids=raw_sam['native_proposal_indices'][original]
        old_area=resized.float().flatten(1).mean(1)
        assert torch.equal(old_native_ids,torch.where((old_area>=.002)&(old_area<=.6))[0])
        assert torch.equal(raw_sam['masks'][original],resized[old_native_ids])
        ids=discarded_small_region_ids(resized)
        assert not set(ids.tolist())&set(old_native_ids.tolist())
        for j in ids.tolist():
            crops.append(pil_to_tensor(original_native_gray_crop(np.asarray(image),native[j])).float()/255)
            masks.append(resized[j]);views.append(vi)
            rows.append(dict(frame=frame,view_index=vi,native_index=j,area=float(old_area[j]),
                quality=float(sam['scores'][j]),split='train' if frame in info['train_frames'] else 'validation',
                rgb_sha256=src['source_sha256'],sam_sha256=sam_hash))
        counts.append(dict(frame=frame,native_regions=len(native),original_retained=len(original),new_small=len(ids)))
    assert len(masks)>0
    print(row['key'],'small native crops',len(crops),flush=True)
    vision=NativeSiglip2Runtime.load(Path(teacher['encoder']['path']),device=torch.device(device))
    assert vision.bundle==teacher['encoder'],'Original native encoder bundle changed'
    desc=[]
    for start in range(0,len(crops),8):desc.append(vision.encode(torch.stack(crops[start:start+8])).cpu().half())
    descriptors=torch.cat(desc)
    del vision,crops;gc.collect();torch.cuda.empty_cache()
    payload=dict(masks=torch.stack(masks),proposal_views=torch.tensor(views),descriptors=descriptors,
        source_rows=rows,encoder=teacher['encoder'],source_only=True,teacher_pseudo_labels=True)
    torch.save(payload,out/'small_teacher.pt')
    (out/'small_inventory.json').write_text(json.dumps(dict(frames=counts,regions=rows,
        small_teacher_sha256=sha256_file(out/'small_teacher.pt'),raw_sam_prefix_replay_exact=True),indent=2)+'\n')
    return payload,counts


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
    small,counts=recover_small(row,info,teacher,bindings,device,out)
    prior=Path('output/native_residual_basis_next30_v2')/key
    mr=json.loads((prior/'map/report.json').read_text());data=load(prior/'map/map.pt',mr['map_sha256'])
    assert data['geometry_binding']==ep['geometry_binding']
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
    numerator={arm:torch.zeros((n,64),device=device) for arm in ('baseline','candidate')}
    masses={arm:torch.zeros(n,device=device) for arm in numerator};fits=[]
    for frame in info['train_frames']:
        vi=info['frames'].index(frame);ids,sid,masks,smasks,pid,gid,w,_=view(vi)
        merged=torch.cat([masks,smasks]);descriptors=torch.cat([compact[ids],small_compact[sid]])
        for arm,mm,cc in [('baseline',masks,compact[ids]),('candidate',merged,descriptors)]:
            lift,_=source_pixel_evidence_weights(mm,pid,gid,w,n,partition_overlap=True)
            numerator[arm]+=lift.T@cc;masses[arm]+=lift.sum(0)
        fits.append(dict(frame=frame,retained_regions=len(ids),appended_small_regions=len(sid)))
        print(key,'fit',frame,len(sid),flush=True)
    replay=numerator['baseline']/masses['baseline'][:,None].clamp_min(1e-8)
    replay_error=float((replay[observed]-features[observed]).abs().max());assert replay_error<2e-5,replay_error
    assert (masses['baseline'][observed]>0).all() and (masses['candidate'][observed]>0).all()
    candidate=features.clone();estimate=numerator['candidate']/masses['candidate'][:,None].clamp_min(1e-8)
    candidate[observed]=estimate[observed]
    assert torch.equal(candidate[~observed],features[~observed]) and torch.isfinite(candidate).all()
    torch.save(dict(features=candidate.cpu(),basis=basis.cpu(),observed=data['observed'],geometry_binding=data['geometry_binding'],
        parent_map_sha256=mr['map_sha256'],source_only=True,old_heads_compatible=False),out/'map.pt')
    pools={arm:{kind:{mode:[] for mode in queries} for kind in ('retained','small','boundary')} for arm in ('baseline','candidate')}
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
        for arm,values in [('baseline',features),('candidate',candidate)]:
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
            b=metrics['baseline'][kind][mode];c=metrics['candidate'][kind][mode]
            gate[mode][kind]=dict(mse_ratio=c['mse']/max(b['mse'],1e-12),top1_drop=b['teacher_top1_agreement']-c['teacher_top1_agreement'])
    admitted=all(g['small']['mse_ratio']<=.95 and g['retained']['mse_ratio']<=1.01 and g['boundary']['mse_ratio']<=1.01
        and g['small']['top1_drop']<=.01 and g['retained']['top1_drop']<=.01 for g in gate.values())
    report=dict(**row,source_only=True,benchmark_data_read=False,source_admitted=admitted,gate=gate,metrics=metrics,
        observations=observations,source_fits=fits,source_counts=counts,new_small_regions=len(small['masks']),
        training_small_regions=sum(r['appended_small_regions'] for r in fits),validation_small_regions=sum(r['small_regions'] for r in observations),
        support_unchanged=True,basis_unchanged=True,geometry_binding=data['geometry_binding'],original_masks_descriptors_changed=False,
        original_heads_reused=False,map_sha256=sha256_file(out/'map.pt'),small_teacher_sha256=sha256_file(out/'small_teacher.pt'),
        baseline_fusion_replay_max_error=replay_error,registration_sha256=sha256_file(OUT/'registration.json'),inputs=bindings,
        caveat='Teacher pseudo-response preservation, not real small-object localization; original boundary pseudo-target still averages overlapping old regions')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(key,admitted,gate,flush=True)


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
