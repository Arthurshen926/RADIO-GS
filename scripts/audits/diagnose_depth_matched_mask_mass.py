"""Matched equal-view versus visibility-mass source-mask transfer.

An exact phrase match is a correspondence hypothesis, not physical-instance GT.
Missing detections are unknown. Target-view masks enter metrics only.
"""
import argparse,json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    assert not a.output.exists()
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    info=json.loads((root/'teachers.json').read_text());original_train_frames=list(info['train_frames']);depth_audit=Path(f'paper/artifacts/{a.scene}_direct_depth_mask_transfer_20260916.json');depth_result=json.loads(depth_audit.read_text());info['train_frames']=[f for f in info['train_frames'] if f not in depth_result['unavailable_calibration_frames']];auth=json.loads((root/'responsibility.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    cp=root/'caption_region_evidence.pt';assert sha256_file(cp)==json.loads(cp.with_suffix('.json').read_text())['sha256'];cap=torch.load(cp,weights_only=False)
    assert cap['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    dev=a.device;n=auth['num_gaussians'];pixels=cap['masks'][0].numel();Q=len(cap['phrases'])
    numerator=torch.zeros(n,Q,device=dev);denominator=torch.zeros_like(numerator);weighted_num=torch.zeros_like(numerator);weighted_den=torch.zeros_like(numerator);cache=[];bindings={}
    with torch.inference_mode():
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);targets={}
            for ri in torch.where(cap['proposal_views']==vi)[0].tolist():
                r=cap['records'][ri];mask=cap['masks'][ri].flatten().to(dev).bool()
                if r['confidence']<.7 or mask.sum()<20:continue
                for qi in r['phrase_indices']:targets[qi]=targets.get(qi,torch.zeros_like(mask))|mask
            if not targets:continue
            qids=sorted(targets);truth=torch.stack([targets[qi] for qi in qids],1)
            rec=auth['views'][vi];path=root/rec['relative_path'];assert sha256_file(path)==rec['sha256'];bindings[rec['relative_path']]=rec['sha256'];hit=torch.load(path,weights_only=False)
            gid,pid,w=[hit[k].to(dev) for k in ['gaussian_ids','pixel_ids','base_weights']];gid=gid.long();pid=pid.long();w=w.float()
            pm=torch.zeros(pixels,device=dev).scatter_add_(0,pid,w);weights=w/pm[pid].clamp_min(1e-8)
            mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weights,(pixels,n)).coalesce()
            em=torch.zeros(n,device=dev).scatter_add_(0,gid,weights);seen=em>1e-6
            lifted=torch.sparse.mm(mat.transpose(0,1),truth.float())/em[:,None].clamp_min(1e-8);lifted*=seen[:,None]
            numerator[:,qids]+=lifted;denominator[:,qids]+=seen[:,None]
            weighted_num[:,qids]+=lifted*em[:,None];weighted_den[:,qids]+=(em*seen)[:,None]
            cache.append((frame,qids,truth,mat,lifted,seen,em))
        counts={qi:sum(qi in row[1] for row in cache) for qi in range(Q)};rows=[]
        for frame,qids,truth,mat,lifted,seen,em in cache:
            count=(denominator[:,qids]-seen[:,None].float()).clamp_min(0)
            other=(numerator[:,qids]-lifted)/count.clamp_min(1)
            support=torch.sparse.mm(mat,(count>0).float())
            score=torch.sparse.mm(mat,other)/support.clamp_min(1e-8)
            pred=(score>.5)&(support>.5)
            weighted_mass=(weighted_den[:,qids]-(em*seen)[:,None]).clamp_min(0)
            weighted_other=(weighted_num[:,qids]-lifted*em[:,None])/weighted_mass.clamp_min(1e-8)
            # Match the original operational support, so only fusion weights change.
            weighted_other=torch.where(count>0,weighted_other,0)
            weighted_score=torch.sparse.mm(mat,weighted_other)/support.clamp_min(1e-8)
            weighted_pred=(weighted_score>.5)&(support>.5)
            same=torch.sparse.mm(mat,lifted)>.5
            for ci,qi in enumerate(qids):
                target=truth[:,ci]
                def iou(p):return float((p&target).sum()/(p|target).sum().clamp_min(1))
                eligible=counts[qi]>1
                rows.append(dict(frame=frame,phrase=cap['phrases'][qi],other_source_views=counts[qi]-1,
                    same_view_iou=iou(same[:,ci]),cross_view_iou=iou(pred[:,ci]) if eligible else None,
                    mass_weighted_iou=iou(weighted_pred[:,ci]) if eligible else None,
                    cross_view_positive_support=float((support[:,ci][target]>.5).float().mean()) if eligible else None))
            print(a.scene,frame,'done',flush=True)
    paired=[r for r in rows if r['cross_view_iou'] is not None]
    macro={k:sum(r[k] for r in paired)/len(paired) for k in ['same_view_iou','cross_view_iou','mass_weighted_iou','cross_view_positive_support']}
    result=dict(schema='depth_matched_source_mask_mass.v1',source_only=True,benchmark_data_used=False,teacher_masks_are_not_gt=True,
        scene=a.scene,train_frames=info['train_frames'],excluded_frames=depth_result['unavailable_calibration_frames'],depth_audit_sha256=sha256_file(depth_audit),observations=len(rows),paired_observations=len(paired),singleton_observations=len(rows)-len(paired),
        no_semantic_features_used=True,no_model_updates=True,missing_detections_unknown=True,target_view_excluded=True,
        protocol='identical caption phrase; union same-view masks confidence>=.7 and area>=20; exact normalized contribution lifting; average other detected views over their visibility only; observed-support normalized source raster; require support>.5 and membership>.5; full target mask denominator',
        macro=macro,rows=rows,caption_sha256=sha256_file(cp),authority_sha256=info['authority_sha256'],view_bindings=bindings,implementation_sha256=sha256_file(__file__))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,macro,flush=True)


if __name__=='__main__':main()
