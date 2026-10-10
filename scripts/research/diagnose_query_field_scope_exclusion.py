"""All admitted source-view fits, no target labels; not a target accuracy bound."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from radio_gs.query_field.relation_training import SourceRelations
from radio_gs.query_field.geometry import relaxed_subset_alpha
from scripts.research.conditional_subset_reference import silhouette_loss
from scripts.research.run_query_field_scope_exclusion import ROOT,read,write,receipt,check,design
from radio_gs.query_field.scope_exclusion import union_exclusion
ARMS=['partial_member','scope_member']


@torch.no_grad()
def main():
    d=design();read(ROOT/'predictions_complete.json');scenes={};teacher_inputs=[]
    frozen=read('output/query_field_20261010/teacher_acceptance_v1/design.json')
    bindings={str(Path(x['path']).resolve()):x for x in frozen['inputs']}
    for j in read(d['configuration']['path'])['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));check(r['teacher_selection'])
        data=torch.load(p/'predictions.pt',weights_only=True);teacher=torch.load(r['teacher_selection']['path'],weights_only=True)
        field=SimpleNamespace(**{k:v.cuda() for k,v in data['geometry'].items()},observed=data['observed'].cuda())
        scale=read(Path(j['field'])/'manifest.json')['spec']['world_units_per_metre']
        source=SourceRelations(j['observations'],field.observed);offset=0;rows=[]
        for vi,(v,original) in enumerate(zip(source.views,read(j['construction'])['source_records'])):
            assert v['frame']==original['frame'] and v['original_regions']==original['regions']
            count=len(v['positive']);accept=teacher['probability'][offset:offset+count]>.5;offset+=count
            raw=torch.load(Path(j['observations'])/f"view_{v['frame']:05d}.pt",weights_only=True);h,w=raw['masks'].shape[-2:];masks=raw['masks'][original['regions']].bool()
            detector_path=Path(j['teacher_detections'])/f"identity_{v['frame']}.pt"
            check(bindings[str(detector_path.resolve())]);teacher_inputs.append(bindings[str(detector_path.resolve())])
            detector=torch.load(detector_path,weights_only=True);assert detector['frame']==v['frame']
            for qi,text in enumerate(data['texts']):
                ids=accept[:,qi]
                if not bool(ids.any()):continue
                pos=v['positive'][ids].any(0).cuda();neg=v['negative'][ids].all(0).cuda();known=pos|neg
                if not bool(pos.any()):continue
                candidate=masks[ids].any(0).flatten().cuda()
                accepted=torch.zeros(len(raw['masks']),dtype=torch.bool)
                accepted[torch.tensor(original['regions'])[ids]]=True
                expanded,_=union_exclusion(raw['masks'].bool(),raw['scales_m'],accepted,
                    raw['raw_mass'].reshape(h,w)>0,neg.cpu().reshape(h,w))
                added=expanded.flatten().cuda() & ~neg
                instance=detector['instances'][detector['queries'].index(text)]
                assert len(instance['scores'])
                target=(F.interpolate(instance['masks'].float().amax(0)[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0]>.5).flatten().cuda()
                controls={}
                for a in ARMS:
                    member=(data['scores'][a][:,qi]>.5).float().cuda()
                    alpha=relaxed_subset_alpha(field,member,v['camera'],h,w,scale);pred=alpha>(10.5/255)
                    tp=int((pred&pos).sum());fp=int((pred&neg).sum());fn=int((~pred&pos).sum())
                    controls[a]=dict(known_iou=tp/max(tp+fp+fn,1),known_precision=tp/max(tp+fp,1),
                        known_recall=tp/max(tp+fn,1),hard_member_source_objective=float(silhouette_loss(alpha,pos,neg)),
                        candidate_union_iou=float((pred&candidate).sum()/(pred|candidate).sum().clamp_min(1)),
                        raw_teacher_iou=float((pred&target).sum()/(pred|target).sum().clamp_min(1)),
                        candidate_to_teacher_iou=float((candidate&target).sum()/(candidate|target).sum().clamp_min(1)),
                        predicted_unknown_fraction=float((pred&~known).sum()/pred.sum().clamp_min(1)),
                        outside_candidate_fraction=float((pred&~candidate).sum()/pred.sum().clamp_min(1)),
                        added_negative_prediction_fraction=float((pred&added).sum()/added.sum().clamp_min(1)))
                rows.append(dict(frame=v['frame'],text=text,positive_pixels=int(pos.sum()),negative_pixels=int(neg.sum()),added_negative_pixels=int(added.sum()),
                    added_negative_teacher_foreground=int((added&target).sum()),controls=controls))
        assert offset==len(teacher['views'])
        expected=sum(q['fitting']['active_views'] for q in r['diagnostics']);assert len(rows)==expected
        mean={a:{k:sum(x['controls'][a][k] for x in rows)/len(rows) for k in rows[0]['controls'][a]} for a in ARMS}
        scenes[j['scene']]=dict(admitted_query_view_pairs=len(rows),mean=mean,rows=rows)
        print(j['scene'],mean,flush=True)
    macro={a:{k:sum(s['mean'][a][k] for s in scenes.values())/len(scenes) for k in next(iter(scenes.values()))['mean'][a]} for a in ARMS}
    permission_totals={k:sum(x[k] for s in scenes.values() for x in s['rows']) for k in ['added_negative_pixels','added_negative_teacher_foreground']}
    report=dict(permission_totals=permission_totals,complete=True,source_only=True,target_labels_opened=False,source_assisted_diagnostic=True,
        hard_alpha_threshold=10.5/255,unknown_pixels_excluded_from_known_metrics_only=True,teacher_inputs=teacher_inputs,macro=macro,scenes=scenes,
        interpretation='Known partial fit and full-raster raw-teacher/candidate agreement on admitted pairs. Teacher is not GT; unknown is not negative. No target labels, heldout claim or selection changes.',code=receipt(__file__))
    write(ROOT/'diagnostics/source_scope.json',report);write('docs/experiments/2026-10-10-scope-exclusion-source-scope.json',report)

if __name__=='__main__':torch.set_num_threads(4);main()
