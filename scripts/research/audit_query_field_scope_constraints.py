"""Full-cohort projected-set scoring and field compilation audit."""
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from scripts.research.run_query_field_scope_constraints import ROOT,FIELDS,SOURCE,ARMS,read,write,receipt,check,design
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();cfg=read(d['configuration']['path']);per=read(ROOT/'summary.json')['per_scene'];details={};labels={};compilation=[]
    for stage in ['predictions','scoring']:
        for r in read(ROOT/(stage+'_complete.json'))['records']:check(r)
    for j in cfg['jobs']:
        compiled=read(FIELDS/j['scene']/'compilation.json');valid=read(FIELDS/j['scene']/'validation.json')
        assert compiled['complete'] and valid['passed'] and valid['parent_tensors_exact']
        parent=read(Path('output/query_field_20261010/compiled_scope_permissions_v1')/j['scene']/'field/manifest.json')
        assert [(x['frame'],x['sha256']) for x in compiled['records']]==[(x['frame'],x['sha256']) for x in parent['provenance']['source_records']]
        assert compiled['field_sha256']==valid['field_sha256']==read(Path(j['field'])/'manifest.json')['files']['field.pt']['sha256']
        assert not compiled['benchmark_queries_used'] and not compiled['target_labels_used']
        assert valid['max_error']<=compiled['tolerance'] and not valid['denied_attempts']
        compilation.append(dict(compilation=compiled,validation=valid))
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']));check(r['teacher_selection'])
        data=torch.load(p/'predictions.pt',weights_only=True);old=torch.load(SOURCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
        assert not r['transient_query_fitting'] and not r['persistent_field_updated'] and not r['target_labels_opened']
        assert not r['runtime_source_raster_access']
        assert torch.equal(data['observed'],old['observed']) and data['texts']==old['texts']
        for k,v in data['geometry'].items():assert torch.equal(v,old['geometry'][k])
        for a in ARMS:
            assert torch.equal(data['scores'][a],old['scores']['measure_prior'])
            assert torch.equal(data['selected_by_control'][a],data['membership_by_control'][a]>.5)
        assert torch.equal(data['selected'],data['selected_by_control']['native_scope'])
        e=read(p/'evaluation/evaluation.json');counts={}
        for provider in ['native','teacher']:
            a=provider+'_partial';b=provider+'_scope'
            counts[provider]=dict(added_member_query_pairs=int((data['selected_by_control'][b]&~data['selected_by_control'][a]).sum()),
                removed_member_query_pairs=int((~data['selected_by_control'][b]&data['selected_by_control'][a]).sum()),
                max_score_increase=float((data['membership_by_control'][b]-data['membership_by_control'][a]).max()))
            assert counts[provider]['max_score_increase']<=2e-6
        for a in ARMS:
            precision=[];recall=[]
            for x in e['metrics'][a]['selected']['objects']:
                assert not x['missing'];gt=load_binary_mask(Path(x['gt_path']),threshold=10)
                pred=load_binary_mask(Path(x['pred_path']),threshold=10,grayscale=True);tp=int((gt&pred).sum())
                assert abs(tp/max(int((gt|pred).sum()),1)-x['iou'])<1e-12
                precision.append(tp/max(int(pred.sum()),1));recall.append(tp/max(int(gt.sum()),1))
                if x['gt_path'] not in labels:labels[x['gt_path']]=receipt(x['gt_path'])
            per[a][j['scene']].update(precision=float(np.mean(precision)),recall=float(np.mean(recall)))
        previous=torch.load(Path('output/query_field_20261010/compositional_atoms_v2')/'predictions'/j['name']/'predictions.pt',weights_only=True)
        error=float((data['membership_by_control']['teacher_partial']-previous['scores']['atom_parts']).abs().max())
        assert error<=2e-6,('partial reference drift',j['scene'],error)
        changed=int((data['selected_by_control']['teacher_partial']!=(previous['scores']['atom_parts']>.5)).sum())
        details[j['scene']]=dict(queries=len(data['texts']),changes=counts,semantic_exact=True,geometry_exact=True,
            partial_reference_max_error=error,partial_reference_changed_member_query_pairs=changed)
    macro={a:{k:sum(x[k] for x in per[a].values())/4 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={provider:{k:paired({s:v[k] for s,v in per[provider+'_partial'].items()},{s:v[k] for s,v in per[provider+'_scope'].items()},'LERF '+k) for k in macro[provider+'_scope']} for provider in ['native','teacher']}
    report=dict(passed=True,macro=macro,per_scene=per,comparisons=comparisons,details=details,compilation=compilation,
        additional_tensor_bytes=sum(x['compilation']['additional_tensor_bytes'] for x in compilation),
        all_predictions_before_scoring=True,gt_files=list(labels.values()),code=receipt(__file__),design=receipt(ROOT/'design.json'),
        teacher_controls_are_privileged=True,model_promotion=False,whole_five_task=False)
    write(ROOT/'validation/audit.json',report);write('docs/experiments/2026-10-10-scope-constraints-audit.json',report);print(macro,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
