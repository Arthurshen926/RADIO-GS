"""Post-hoc actual-raster extension attribution. Evaluation only; no fitting."""
import json
from pathlib import Path
import numpy as np
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from radio_gs.query_field.storage import sha256

ROOT=Path('output/query_field_20261011/member_change_factorial_v1')
OUT=Path('output/query_field_20261011/extension_quality_audit_v1')


def compare_masks(initial, expanded, target):
    if initial.dtype != bool or expanded.dtype != bool or target.dtype != bool or initial.shape != expanded.shape or initial.shape != target.shape:
        raise ValueError('aligned binary masks required')
    added=expanded & ~initial; removed=initial & ~expanded
    tp=int((initial & target).sum()); union=int((initial | target).sum())
    gain=int((added & target).sum()); leak=int((added & ~target).sum())
    lost=int((removed & target).sum()); repaired=int((removed & ~target).sum())
    before=tp/max(union,1);after=int((expanded & target).sum())/max(int((expanded | target).sum()),1)
    # Complete exact counterfactual identity, including non-monotone renderers.
    replay=(tp+gain-lost)/max(union+leak-repaired,1)
    if abs(replay-after)>1e-12:raise AssertionError('counterfactual decomposition differs')
    return dict(initial_iou=before,expanded_iou=after,delta_iou=after-before,
        initial_tp=tp,initial_union=union,added_tp=gain,added_fp=leak,removed_tp=lost,removed_fp=repaired,
        added_pixels=gain+leak,removed_pixels=lost+repaired,
        added_precision=gain/(gain+leak) if gain+leak else None,
        # For monotone expansion, improve iff added precision > IoU/(1+IoU).
        break_even_added_precision=before/(1+before),
        expanded_precision=int((expanded & target).sum())/max(int(expanded.sum()),1),
        beneficial=after>before, harmful=after<before)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    scenes={};receipts=[]
    for path in sorted((ROOT/'predictions').glob('*/evaluation/evaluation.json')):
        e=json.loads(path.read_text()); rows=[]
        receipts.append(dict(path=str(path),sha256=sha256(path)))
        def indexed(arm):
            objects=e['metrics'][arm]['selected']['objects']
            result={(x['frame'],x['query']):x for x in objects}
            assert len(result)==len(objects) and all(not x['missing'] for x in objects)
            return result
        initial=indexed('semantic_initial');expanded=indexed('addition_only')
        assert initial.keys()==expanded.keys()
        for key,a in initial.items():
            b=expanded[key];assert a['gt_path']==b['gt_path']
            gt=load_binary_mask(Path(a['gt_path']),threshold=10)
            i=load_binary_mask(Path(a['pred_path']),threshold=10,grayscale=True)
            s=load_binary_mask(Path(b['pred_path']),threshold=10,grayscale=True)
            r=compare_masks(i,s,gt)
            assert abs(r['initial_iou']-a['iou'])<1e-12 and abs(r['expanded_iou']-b['iou'])<1e-12
            rows.append(dict(frame=key[0],query=key[1],**r))
        changed=[r for r in rows if r['added_pixels']]
        total={k:sum(r[k] for r in rows) for k in ['added_tp','added_fp','removed_tp','removed_fp','added_pixels','removed_pixels']}
        by_query={q:[r for r in rows if r['query']==q] for q in sorted({r['query'] for r in rows})}
        scenes[e['job']['scene']]=dict(cases=len(rows),changed_cases=len(changed),totals=total,
            pixel_micro_added_precision=total['added_tp']/max(total['added_pixels'],1),
            changed_case_macro_added_precision=float(np.mean([r['added_precision'] for r in changed])) if changed else None,
            harmful_cases=sum(r['harmful'] for r in rows),beneficial_cases=sum(r['beneficial'] for r in rows),
            core_hides_bad_extension_cases=sum(r['expanded_precision']>.5 and r['harmful'] for r in rows),
            mean_delta_iou=float(np.mean([r['delta_iou'] for r in rows])),
            queries={q:dict(mean_delta_iou=float(np.mean([r['delta_iou'] for r in rs])),
                           added_tp=sum(r['added_tp'] for r in rs),added_fp=sum(r['added_fp'] for r in rs)) for q,rs in by_query.items()},rows=rows)
        print(e['job']['scene'],{k:v for k,v in scenes[e['job']['scene']].items() if k not in ['rows','queries']},flush=True)
    assert len(scenes)==4
    report=dict(post_hoc=True,evaluation_only=True,no_new_predictions=True,model_promotion=False,
        independent_final_test=False,input_reports=receipts,code_sha256=sha256(__file__),scenes=scenes,
        all_ious_replayed=True,macro_delta_iou=float(np.mean([r['mean_delta_iou'] for r in scenes.values()])),
        note='Pixel changes of actual selected-set renders; not unique Gaussian truth or 3D IoU. Break-even threshold uses GT for diagnosis only; forbidden as inference input.')
    text=json.dumps(report,indent=2)+'\n';(OUT/'audit.json').write_text(text)
    Path('docs/experiments/2026-10-11-extension-quality-audit.json').write_text(text)

if __name__=='__main__':main()
