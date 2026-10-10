"""Evidence support of fixed teacher interventions; no target label access.

Gaussian counts are not projected error contributions or identity ground truth.
Replays frozen acceptance; never changes scores or selections for evaluation.
"""
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.part_conditioning import prepare_parts,read_parts,integrate_views
from radio_gs.query_field.region_union import prepare_union_parts
from scripts.research.run_query_field_teacher_acceptance import ROOT,SOURCE,read,write,receipt,check,design


@torch.no_grad()
def main():
    d=design(verify_inputs=False);read(ROOT/'scoring_complete.json');scenes={}
    names=['unknown','positive_only','negative_only','conflicting']
    for job in read(d['configuration']['path'])['jobs']:
        out=ROOT/'predictions'/job['name'];record=read(out/'prediction.json')
        check(record['teacher_selection'])
        check(dict(path=str(out/'predictions.pt'),sha256=record['predictions_sha256']))
        data=torch.load(out/'predictions.pt',weights_only=True)
        prior_data=torch.load(SOURCE/'predictions'/job['name']/'predictions.pt',weights_only=True)
        for r in d['inputs']:
            if r['path']==str(SOURCE/'predictions'/job['name']/'predictions.pt'):check(r)
        field,_=load_field(job['field'],'cuda');available=field.observed|field.semantic_measure.supported
        probability=torch.load(out/'teacher_selection.pt',weights_only=True)['probability'].cuda();rows=[]
        for qi,text in enumerate(data['texts']):
            prior=prior_data['scores']['measure_prior'][:,qi].cuda();arms={}
            for prefix,prepare in [('teacher',prepare_parts),('teacher_union',prepare_union_parts)]:
                prepared=prepare(field.evidence,prior,available,region_probability=probability[:,qi])
                positive,negative,_=prepared;p=positive.sum(1)
                for suffix,protect in [('parts',False),('protected',True)]:
                    arm=prefix+'_'+suffix;scores,_=read_parts(prior,available,prepared,protect=protect)
                    _,exclusion=integrate_views(prior,positive,negative,protect=protect);n=exclusion.sum(1)
                    saved=data['scores'][arm][:,qi].cuda();selected=scores>.5
                    error=float((saved-scores).abs().max())
                    assert error<=1e-6 and torch.equal(selected,data['selected_by_control'][arm][:,qi].cuda())
                    group=(p>0).long()+2*(n>0).long()
                    total=torch.bincount(group[available],minlength=4)
                    chosen=torch.bincount(group[selected],minlength=4)
                    assert bool(torch.equal(selected[(group==0)&available],(prior>.5)[(group==0)&available]))
                    arms[arm]=dict(available_by_evidence=dict(zip(names,total.cpu().tolist())),
                        selected_by_evidence=dict(zip(names,chosen.cpu().tolist())),
                        selected_without_positive_fraction=float(((p==0)&selected).sum()/selected.sum().clamp_min(1)),
                        score_replay_max_error=error,selection_replay_exact=True)
            rows.append(dict(text=text,controls=arms))
        scenes[job['scene']]=rows
        print(job['scene'],'support replay complete',flush=True)
        del field,data,prior_data
    arms=list(next(iter(scenes.values()))[0]['controls'])
    per_scene={s:{a:sum(r['controls'][a]['selected_without_positive_fraction'] for r in rows)/len(rows) for a in arms} for s,rows in scenes.items()}
    report=dict(complete=True,target_labels_opened=False,new_predictions=False,source_assisted_diagnostic=True,
        interpretation='Nonzero evidence may be weak; Gaussian count shares are not pixel error or causal recovery estimates.',
        scenes=scenes,per_scene_without_positive_fraction=per_scene,
        macro_without_positive_fraction={a:sum(x[a] for x in per_scene.values())/len(per_scene) for a in arms},
        code=receipt(__file__),design=receipt(ROOT/'design.json'))
    write(ROOT/'diagnostics/support.json',report)
    write('docs/experiments/2026-10-10-teacher-support-diagnostic.json',report)


if __name__=='__main__':torch.set_num_threads(4);main()
