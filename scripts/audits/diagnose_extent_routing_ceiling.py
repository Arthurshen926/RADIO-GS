"""GT-selected extent capacity diagnostic, never a deployable retrieval score."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    p.add_argument('--ungrouped',action='store_true')
    p.add_argument('--region-head',action='store_true')
    p.add_argument('--restricted',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    suffix='extent_region_oracle' if a.ungrouped else 'extent_capacity_oracle'
    if a.region_head:suffix='learned_region_oracle'
    if a.restricted:
        assert a.region_head
        suffix='region_selection_oracle'
    output=Path(f'paper/artifacts/{a.scene}_{suffix}_20260916.json')
    if output.exists():raise FileExistsError(output)
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output');root=base/'preserved_source_inventory_20260916'/a.scene
    report=json.loads((root/'grounded_semantic_transfer96/report.json').read_text());groups=report['groups']
    path=root/'caption_region_evidence.pt';assert sha256_file(path)==report['evidence_sha256'];e=torch.load(path,weights_only=False)
    gr=json.loads((base/'geometry_only_repair_final_20260914'/a.scene/'report.json').read_text());g=gr['results']['independent_scale']
    geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
    prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);identity=json.loads((root/'native_query_encoder.json').read_text())
    ann,categories,sh,sw=load_labels(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'),a.scene)
    names=['source_region_extent96'] if a.region_head else ['grounded_semantic_transfer96','source_extent_rendered_residual96']
    responses=[];digests={};labels=[]
    queries=NativeThreeQueryEncoder(identity,a.device).encode_text(categories).half().float() if a.restricted else None
    fallback={}
    if a.restricted:
        baseline=json.loads(Path(f'paper/artifacts/{a.scene}_caption_transfer_text_20260916.json').read_text())['runs']['full_source_query_threshold_control']
        fallback={(r['frame'],r['category']):r['iou'] for r in baseline['observations']}
    with torch.no_grad():
        for name in names:
            path=root/name/'field.pt';digest=json.loads((path.parent/'report.json').read_text())['field_sha256']
            field,_=load_element_retrieval_field(path,expected_sha256=digest,geometry=geo,expected_query_encoder=identity,device=a.device)
            head=field.grounded_transfer;features=head.features(field.latent);raw=head.decoder(features)
            if head.residual is not None:raw+=head.residual(features)
            responses.append(raw.clamp(1e-6,1-1e-6)*field.observed[:,None]);digests[name]=digest
            if head.num_proposals:
                indices=json.loads((path.parent/'report.json').read_text())['proposal_indices']
                labels.append([f"region {i}, frame {e['records'][i]['frame']}" for i in indices])
                if a.restricted:
                    route,accepted,_=head.route(queries);eligible=head.proposal_groups[:,route]
                    accepted=accepted&eligible.any(0)
            else:labels.append([g['phrase'] for g in groups])
        observed=field.observed;prob=e['region_probability'].float().to(a.device);seen=e['seen'].to(a.device);memberships=[]
        for group in groups:
            numerator=torch.zeros(len(observed),device=a.device);denominator=torch.zeros_like(numerator)
            for vi in group['views']:
                ids=[i for i in group['regions'] if int(e['proposal_views'][i])==vi]
                numerator+=prob[ids].amax(0);denominator+=seen[vi]
            memberships.append(numerator/denominator.clamp_min(1)*observed)
        names.append('uncompressed_source_membership_diagnostic');responses.append(torch.stack(memberships,1))
        labels.append([g['phrase'] for g in groups])
        if a.ungrouped:
            names.append('individual_source_region_diagnostic');responses.append(prob.T*observed[:,None])
            labels.append([f"region {i}, frame {r['frame']}" for i,r in enumerate(e['records'])])
        colors=torch.cat(responses+[observed.float()[:,None]],1);width=480;height=round(sh*width/sw);records=[]
        for frame,objects in sorted(ann.items()):
            camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=width/camera.width;k[1]*=height/camera.height
            image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.reshape(-1),colors=colors,
                viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=width,height=height,
                near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            predictions=((image[0,...,:-1]/mass[...,None].clamp_min(1e-8)>.5)&valid[...,None]).flatten(0,1)
            targets=ground_truth_masks(objects,categories,height,width,sh,sw)
            for c in categories:
                target=targets[c].to(a.device).flatten()
                if not target.any():continue
                ious=(predictions&target[:,None]).sum(0)/(predictions|target[:,None]).sum(0).clamp_min(1)
                row=dict(frame=frame,category=c)
                offset=0
                for i,name in enumerate(names):
                    score,index=ious[offset:offset+len(labels[i])].max(0);offset+=len(labels[i])
                    row[name]=dict(iou=float(score),source_phrase=labels[i][int(index)])
                if a.restricted:
                    ci=categories.index(c)
                    if accepted[ci]:
                        score,index=ious[:len(labels[0])].masked_fill(~eligible[:,ci],-1).max(0)
                        row['concept_restricted_region_oracle']=dict(iou=float(score),source_phrase=labels[0][int(index)])
                    else:row['concept_restricted_region_oracle']=dict(iou=fallback[frame,c],source_phrase='exact native fallback')
                records.append(row)
        result=dict(scene=a.scene,diagnostic_only=True,target_ground_truth_selects_head=True,semantic_compatibility_not_required=True,
            deployable_result=False,geometry_sha256=g['geometry_sha256'],fields=digests,evidence_sha256=report['evidence_sha256'],
            limitation='per-observation best mask among source groups; ignores query semantics; source membership is uncompressed diagnostic state',
            observations=records,macro={name:sum(r[name]['iou'] for r in records)/len(records) for name in names},implementation_sha256=sha256_file(__file__))
        if a.restricted:result['macro']['concept_restricted_region_oracle']=sum(r['concept_restricted_region_oracle']['iou'] for r in records)/len(records)
        output.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,result['macro'],flush=True)


if __name__=='__main__':main()
