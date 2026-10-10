"""All admitted source-view fits, no target labels; not a target accuracy bound."""
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.query_field.relation_training import SourceRelations
from radio_gs.query_field.geometry import relaxed_subset_alpha
from scripts.research.conditional_subset_reference import silhouette_loss
from scripts.research.run_query_field_joint_subset_reference import ROOT,ARMS,read,write,receipt,check,design


@torch.no_grad()
def main():
    d=design();read(ROOT/'predictions_complete.json');scenes={}
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
            raw=torch.load(Path(j['observations'])/f"view_{v['frame']:05d}.pt",weights_only=True);h,w=raw['masks'].shape[-2:];del raw
            for qi,text in enumerate(data['texts']):
                ids=accept[:,qi]
                if not bool(ids.any()):continue
                pos=v['positive'][ids].any(0).cuda();neg=v['negative'][ids].all(0).cuda();known=pos|neg
                if not bool(pos.any()):continue
                controls={}
                for a in ARMS:
                    member=(data['scores'][a][:,qi]>.5).float().cuda()
                    alpha=relaxed_subset_alpha(field,member,v['camera'],h,w,scale);pred=alpha>(10.5/255)
                    tp=int((pred&pos).sum());fp=int((pred&neg).sum());fn=int((~pred&pos).sum())
                    controls[a]=dict(known_iou=tp/max(tp+fp+fn,1),known_precision=tp/max(tp+fp,1),
                        known_recall=tp/max(tp+fn,1),hard_member_source_objective=float(silhouette_loss(alpha,pos,neg)))
                rows.append(dict(frame=v['frame'],text=text,positive_pixels=int(pos.sum()),negative_pixels=int(neg.sum()),controls=controls))
        assert offset==len(teacher['views'])
        expected=sum(q['controls']['joint_footprint']['active_views'] for q in r['diagnostics']);assert len(rows)==expected
        mean={a:{k:sum(x['controls'][a][k] for x in rows)/len(rows) for k in rows[0]['controls'][a]} for a in ARMS}
        scenes[j['scene']]=dict(admitted_query_view_pairs=len(rows),mean=mean,rows=rows)
        print(j['scene'],mean,flush=True)
    macro={a:{k:sum(s['mean'][a][k] for s in scenes.values())/len(scenes) for k in next(iter(scenes.values()))['mean'][a]} for a in ARMS}
    report=dict(complete=True,source_only=True,target_labels_opened=False,source_assisted_diagnostic=True,
        hard_alpha_threshold=10.5/255,unknown_pixels_excluded=True,macro=macro,scenes=scenes,
        interpretation='Seen-source partial-label agreement, not heldout source fit, target performance, global optimum or generalization.',code=receipt(__file__))
    write(ROOT/'diagnostics/source_fit.json',report);write('docs/experiments/2026-10-10-joint-subset-source-fit.json',report)

if __name__=='__main__':torch.set_num_threads(4);main()
