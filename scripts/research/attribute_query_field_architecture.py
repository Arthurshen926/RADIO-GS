"""Evaluation-only attribution of the fixed full-cohort architecture baseline."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from radio_gs.query_field.storage import load_field, sha256
from radio_gs.query_field.contracts import Support

BASE=Path('output/query_field_20261007/five_benchmark_v1')
OUT=Path('output/query_field_20261008/architecture_attribution_v1')
COHORT=Path('configs/query_field/five_benchmark_v1/cohort.json')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def binary(selected,target,observed=None):
    selected=np.asarray(selected,dtype=bool);target=np.asarray(target,dtype=bool)
    tp=int((selected&target).sum());fp=int((selected&~target).sum());fn=int((~selected&target).sum())
    result=dict(tp=tp,fp=fp,fn=fn,iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
    if observed is not None:
        observed=np.asarray(observed,dtype=bool)
        result.update(target_observed_fraction=float(observed[target].mean()),
                      fn_unobserved=int((target&~selected&~observed).sum()),
                      fn_observed=int((target&~selected&observed).sum()),
                      observed_target_recall=tp/max(int((target&observed).sum()),1))
        assert result['fn_observed']+result['fn_unobserved']==fn
    return result


def macro(rows,keys):
    scenes=sorted({r['scene'] for r in rows})
    return {k:float(np.mean([np.mean([r[k] for r in rows if r['scene']==s]) for s in scenes])) for k in keys}


def field_jobs(cohort):
    jobs=[dict(family=j['family'],scene=j['name'],field=j['field']) for j in cohort['construction']]
    jobs += [dict(family='lerf',scene=j['scene'],field=j['field']) for j in cohort['lerf']]
    jobs.append(dict(family='nvos',scene='fern',field='output/query_field_20261007/nvos_fern_v1/models/trained'))
    return jobs


def inventory(cohort,device):
    rows=[]
    for j in field_jobs(cohort):
        root=Path(j['field']);m=read(root/'manifest.json');t=read(root.parent/'training.json')
        episodes=t['training_episodes'];visits=t['unique_training_episodes_sampled']
        rows.append(dict(**j,elements=m['elements'],observed_fraction=m['observed']/m['elements'],
            eligible_fraction=m['eligible_anchors']/m['elements'],default_scale_m=m['spec']['default_scale'],
            episodes=episodes,unique_sampled=visits,episode_coverage=visits/episodes,
            updates_per_episode=t['config']['steps']/episodes,support_counts=t.get('support_counts'),
            render_objective=t['render_objective'],fixed_partial_loss_before=t['fixed_training_partial_loss_before'],
            fixed_partial_loss_after=t['fixed_training_partial_loss_after'],field_sha256=m['files']['field.pt']['sha256']))
    write(OUT/'inventory.json',dict(fields=rows,count=len(rows),no_training=True))


def scannet(cohort,device):
    from plyfile import PlyData
    from radio_gs.scannet_constants import compute_split_metrics
    rows=[]
    for job in cohort['scannet']:
        root=BASE/'predictions/scannet'/job['name'];r=read(root/'prediction.json')
        assert sha256(root/'predictions.pt')==r['predictions_sha256']
        p=torch.load(root/'predictions.pt',map_location='cpu',weights_only=True)
        e=read(job['evaluation']);old=read(root/'evaluation.json')
        gt=PlyData.read(e['ground_truth'])['vertex'].data
        xyz=np.stack([gt[k] for k in ['x','y','z']],1);truth=np.asarray(gt['label'])
        dist,idx=cKDTree(p['xyz'].numpy()).query(xyz,workers=4)
        obs=p['observed'].numpy()[idx];splits={}
        for split,ids in e['splits'].items():
            columns=[next(i for i,q in enumerate(p['queries']) if q['class_id']==c) for c in ids]
            scores={a:p[a].numpy()[idx][:,columns] for a in ['native','member']}
            labels={a:np.where(obs,np.asarray(ids)[v.argmax(1)],0) for a,v in scores.items()}
            metrics={a:compute_split_metrics(v,truth,ids) for a,v in labels.items()}
            for a in metrics:assert abs(metrics[a]['miou']-old['metrics'][a][split]['miou'])<1e-12
            valid=np.isin(truth,ids);seen=valid&obs
            n=labels['native']==truth;m=labels['member']==truth
            sorted_scores=np.sort(scores['member'][seen],axis=1)
            ceiling=compute_split_metrics(np.where(obs,truth,0),truth,ids)['miou']
            splits[split]=dict(native_miou=metrics['native']['miou'],member_miou=metrics['member']['miou'],
                perfect_observed_label_miou=ceiling,valid_points=int(valid.sum()),observed_valid_fraction=float(obs[valid].mean()),
                both_correct=int((seen&n&m).sum()),native_correct_member_wrong=int((seen&n&~m).sum()),
                native_wrong_member_correct=int((seen&~n&m).sum()),both_wrong_observed=int((seen&~n&~m).sum()),
                all_member_below_half_fraction=float((sorted_scores[:,-1]<=.5).mean()),
                member_top_tie_fraction=float((sorted_scores[:,-1]==sorted_scores[:,-2]).mean()),
                member_top_margin_median=float(np.median(sorted_scores[:,-1]-sorted_scores[:,-2])),
                native_per_class=metrics['native']['per_class'],member_per_class=metrics['member']['per_class'])
        rows.append(dict(scene=job['name'],splits=splits,max_transport_distance=float(dist.max())))
        print('scannet',job['name'],flush=True)
    write(OUT/'scannet.json',dict(scenes=rows,metrics_replayed=True,privileged_coverage_ceiling=True))


@torch.no_grad()
def uqis(cohort,device):
    package=Path(cohort['uqis_package']);targets=read(package/'evaluator/targets.json')
    root=BASE/'predictions/uqis';evaluation=read(root/'evaluation.json')
    baseline={r['query_id']:r for r in evaluation['explicit_selection']['per_query']}
    for job in cohort['uqis']:
        out=OUT/'uqis'/f"{job['name']}.json"
        if out.exists():continue
        field,manifest=load_field(job['field'],device)
        domain=next(d for d in targets['scene_domains'] if d['scene_id']==job['name'])
        xyz=np.load(package/'method'/domain['mesh_xyz_path']);ids=np.load(package/'evaluator'/domain['mesh_instance_ids_path'])
        dist,index=cKDTree(field.xyz.cpu().numpy()).query(xyz,workers=4)
        gdist,gindex=cKDTree(xyz).query(field.xyz.cpu().numpy(),workers=4)
        gid=ids[gindex];known=gdist<=1e-6
        observed=field.observed.cpu().numpy()[index]
        index_gpu=torch.from_numpy(index).to(device);rows=[]
        receipts={r['query_id']:r for r in read(root/'receipts'/f"{job['name']}.json")['queries']}
        for target in [t for t in targets['targets'] if t['scene_id']==job['name']]:
            truth=ids==target['instance_id']
            for modality,qid in target['queries'].items():
                saved=torch.load(root/'gaussians'/f'{qid}.pt',map_location='cpu',weights_only=True)
                assert sha256(root/'gaussians'/f'{qid}.pt')==receipts[qid]['gaussian_sha256']
                selected=np.load(root/'selections'/f'{qid}.npy');metrics=binary(selected,truth,observed)
                assert abs(metrics['iou']-baseline[qid]['iou'])<1e-12
                assert np.array_equal(saved['selected'].numpy()[index],selected)
                candidates=saved['candidates'].long().numpy();weights=saved['candidate_weights'].numpy()
                row=dict(scene=job['name'],query_id=qid,modality=modality,instance_id=target['instance_id'],
                         status=receipts[qid]['status'],**metrics)
                if len(candidates):
                    association=(gid[candidates]==target['instance_id'])&known[candidates]
                    row.update(support_target_mass=float(weights[association].sum()/weights.sum()),
                               support_unknown_mass=float(weights[~known[candidates]].sum()/weights.sum()),
                               top_candidate_target=bool(association[weights.argmax()]),
                               any_candidate_target=bool(association.any()))
                if modality in {'text','image'}:
                    alternatives=[]
                    for k,c in enumerate(candidates):
                        support=Support(torch.tensor([c],device=device),torch.ones(1,device=device))
                        scores=field.member_logits(support,field.default_scale).sigmoid().masked_fill(~field.observed,0)
                        if k==int(weights.argmax()):
                            error=float((scores.cpu()-saved['membership']).abs().max())
                            assert error<=1e-6,error
                            assert torch.equal(scores.cpu()>.5,saved['selected'])
                            row['replay_max_abs_error']=error
                        values=binary((scores[index_gpu]>.5).cpu().numpy(),truth,observed)
                        alternatives.append(dict(candidate=int(c),associated_target=bool(association[k]),**values))
                    row.update(alternatives=alternatives,oracle_candidate_iou=max(a['iou'] for a in alternatives))
                rows.append(row)
        write(out,dict(scene=job['name'],field_sha256=manifest['files']['field.pt']['sha256'],queries=rows,
                       oracle='GT selection among existing eight candidates at unchanged scale; diagnostic only',
                       gaussian_annotation_max_distance=float(gdist.max())))
        print('uqis',job['name'],flush=True)
        del field;gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def nvos(cohort,device):
    import cv2
    from radio_gs.query_field.public_nvos import load_strokes,retained_response
    from radio_gs.query_field.adapters import scribble_packet
    from radio_gs.query_field.geometry import projection
    for job in cohort['nvos']:
        out=OUT/'nvos'/f"{job['name']}.json"
        if out.exists():continue
        config=read(job['prediction_config']);query=config['jobs'][0]
        field,manifest=load_field(job['field'],device)
        saved_root=BASE/'predictions/nvos'/job['name']/job['name']
        receipt=read(saved_root/'prediction.json');saved=torch.load(saved_root/'predictions.pt',map_location='cpu',weights_only=True)
        assert sha256(saved_root/'predictions.pt')==receipt['predictions_sha256']
        fixture=torch.load(query['camera_fixture'],map_location='cpu',weights_only=True)
        reference=SimpleNamespace(**fixture['reference']);target_camera=SimpleNamespace(**fixture['target'])
        h,w=config['projection_hw'];scale=field.default_scale
        masks,_=load_strokes(query['positive'],query['negative'],(h,w))
        geometry=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,
            get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
        _,op,_,_=projection(geometry,reference,h,w,field.world_units_per_metre)
        packet,_=scribble_packet(op,masks[0],masks[1])
        positive=field.support_logits(packet.positive,scale).sigmoid()
        signed=field.exclude_membership(positive,packet.negative,scale).masked_fill(~field.observed,0)
        negative=1-field.exclude_membership(torch.ones_like(positive),packet.negative,scale)
        error=float((signed.cpu()-saved['membership']).abs().max());assert error<=1e-6,error
        assert torch.equal(signed.cpu()>.5,saved['selected'])
        positive=positive.masked_fill(~field.observed,0)
        rh,rw=config['output_hw']
        positive_response=retained_response(field,positive,target_camera,rh,rw,field.world_units_per_metre).cpu().numpy()
        scoring=read(job['scoring_config']);raw=np.asarray(Image.open(scoring['gt_mask']))
        truth=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
        responses={'positive_only':positive_response,'signed':saved['retained_response'].numpy(),'actual_subset':saved['response'].numpy()}
        metrics={};quantiles={}
        for name,array in responses.items():
            resized=cv2.resize(array,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)
            metrics[name]=binary(resized>.5 if name=='actual_subset' else resized>=.5,truth)
            quantiles[name]=dict(target=np.quantile(resized[truth],[.1,.5,.9]).tolist(),background=np.quantile(resized[~truth],[.1,.5,.9]).tolist())
        original=read(saved_root.parent/'evaluation.json')['results'][0]
        assert abs(metrics['signed']['iou']-original['retained_response']['foreground_iou'])<1e-12
        assert abs(metrics['actual_subset']['iou']-original['foreground_iou'])<1e-12
        ref_pos=torch.sparse.mm(op,positive[:,None])[:,0].cpu();ref_signed=torch.sparse.mm(op,signed[:,None])[:,0].cpu()
        ref_neg=torch.sparse.mm(op,negative[:,None])[:,0].cpu()
        prompt={}
        for sign,mask in zip(['positive','negative'],masks):
            weight=mask.flatten();prompt[sign]={name:float((value*weight).sum()/weight.sum()) for name,value in [('positive_component',ref_pos),('negative_component',ref_neg),('signed',ref_signed)]}
        write(out,dict(scene=job['name'],field_sha256=manifest['files']['field.pt']['sha256'],replay_max_abs_error=error,
            metrics=metrics,response_quantiles=quantiles,prompt_response=prompt,
            positive_selected=int((positive>.5).sum()),signed_selected=int((signed>.5).sum()),
            diagnostic_only=True,negative_removal_not_promoted=True,scale_m=scale))
        print('nvos',job['name'],flush=True)
        del field;gc.collect();torch.cuda.empty_cache()


def lerf(cohort,device):
    from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
    rows=[]
    for job in cohort['lerf']:
        result=read(Path(job['predictions'])/'evaluation/evaluation.json')['metrics']['shared_membership']['selected']
        for item in result['objects']:
            assert not item['missing']
            gt=load_binary_mask(Path(item['gt_path']),threshold=10)
            selected=load_binary_mask(Path(item['pred_path']),threshold=10,grayscale=True)
            metrics=binary(selected,gt);assert abs(metrics['iou']-item['iou'])<1e-12
            rows.append(dict(scene=job['scene'],query=item['query'],frame=item['frame'],**metrics))
    write(OUT/'lerf.json',dict(query_views=rows,scene_macro=macro(rows,['iou','precision','recall']),iou_replayed=True))


@torch.no_grad()
def fusion(cohort,device):
    from sklearn.metrics import average_precision_score
    from radio_gs.query_field.observations import partial_mask_target
    for job in field_jobs(cohort):
        out=OUT/'fusion'/f"{job['family']}_{job['scene']}.json"
        if out.exists():continue
        field,manifest=load_field(job['field'],device)
        obsroot=Path(job['field']).parents[1]/'observations';obs=read(obsroot/'observations.json');frames=[]
        for frame in obs['splits']['development_frames']:
            entry=next(r for r in obs['frames'] if r['frame']==frame)
            assert sha256(obsroot/f'view_{frame:05d}.pt')==entry['compiled']['sha256']
            view=torch.load(obsroot/f'view_{frame:05d}.pt',map_location='cpu',weights_only=True)
            op=view['operator'].to(device);mass=torch.sparse.mm(op,field.observed.float()[:,None])[:,0]
            metrics={};rendered={}
            for name in ['semantic','local']:
                source=F.normalize(view[name+'_pixels'].to(device).float(),dim=-1)
                values=torch.cat([torch.sparse.mm(op,codes.float())
                                  for codes in getattr(field,name).codes.split(128,dim=1)],dim=1)
                valid=(mass>.5)&(source.norm(dim=-1)>0)
                cosine=(F.normalize(values,dim=-1)*source).sum(-1)
                metrics[name]=dict(valid_pixels=int(valid.sum()),cosine_mean=float(cosine[valid].mean()),
                                   cosine_median=float(cosine[valid].median()))
                rendered[name]=values
            desc=F.normalize(view['semantic_descriptors'].to(device).float(),dim=-1)
            teacher=F.normalize(view['semantic_pixels'].to(device).float(),dim=-1)@desc.T
            # Gaussian native scoring then linear projection, with the same observed normalization.
            projected=[]
            for queries in desc.split(16):
                field_scores=field.semantic.score(queries)
                field_scores*=field.observed[:,None]
                projected.append(torch.sparse.mm(op,field_scores)/mass[:,None].clamp_min(1e-8))
            projected=torch.cat(projected,dim=1)
            regions=[]
            for k in range(len(desc)):
                pos,neg=partial_mask_target(view['masks'],k,view['scales_m'])
                visible=(view['raw_mass']>0).reshape(pos.shape);pos&=visible;neg&=visible
                known=(pos|neg).flatten();truth=pos.flatten()[known].numpy()
                if not pos.any() or not neg.any():continue
                regions.append(dict(region=k,positive_pixels=int(pos.sum()),negative_pixels=int(neg.sum()),
                    teacher_ap=float(average_precision_score(truth,teacher[:,k].cpu().numpy()[known])),
                    field_ap=float(average_precision_score(truth,projected[:,k].cpu().numpy()[known])),
                    observed_positive_fraction=float((mass.cpu().reshape(pos.shape)[pos]>.5).float().mean())))
            frames.append(dict(frame=frame,native_direction=metrics,regions=regions))
        write(out,dict(**job,field_sha256=manifest['files']['field.pt']['sha256'],frames=frames,
            source_teacher_only=True,development_views_excluded_from_fusion=True,
            interpretation='Native fusion plus geometry/projection retention, not an isolated fusion causal effect'))
        print('fusion',job['family'],job['scene'],flush=True)
        del field;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['inventory','scannet','uqis','nvos','lerf','fusion'])
    parser.add_argument('--device',default='cuda');args=parser.parse_args();torch.set_num_threads(4)
    assert (BASE/'scoring_complete.json').exists()
    cohort=read(COHORT);globals()[args.stage](cohort,args.device)
