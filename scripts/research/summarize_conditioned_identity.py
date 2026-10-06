"""Frozen evaluation summary; no score-based routing or checkpoint selection."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    stages={'fit':Path('output/scope_evidence_control/fit'),'identity_veto':Path('output/scope_evidence_control/identity_veto'),'conditioned':root/'conditioned','forced_split':root/'forced_split','full_reference':Path('output/research_consolidation/source_run'),'full_conditioned':root/'full_source'}
    arms={}
    for arm,base in stages.items():
        scenes=[];query_rows=[]
        for scene in ['figurines','ramen','teatime','waldo_kitchen']:
            evaluation=json.loads((base/scene/'evaluation/evaluation.json').read_text())['metrics']['hierarchical_pipeline']
            fit=json.loads((base/scene/'sets'/f'lerf_{scene}'/'fit.json').read_text());held=[r for r in fit.get('heldout_teacher_agreement',[]) if r['views']]
            scenes.append(dict(scene=scene,selected_iou=evaluation['selected']['miou'],semantic_iou=evaluation['iou'],ap=evaluation['ap'],heldout_teacher_iou=float(np.mean([h['mean_view_iou'] for h in held])) if held else None,fit_seconds=fit['seconds']))
            objects=evaluation['selected']['objects']
            for query in sorted(set(o['query'] for o in objects)):
                query_rows.append(dict(scene=scene,query=query,iou=float(np.mean([o['iou'] for o in objects if o['query']==query]))))
        arms[arm]=dict(scenes=scenes,macro={k:float(np.mean([s[k] for s in scenes if s[k] is not None])) if any(s[k] is not None for s in scenes) else None for k in ['selected_iou','semantic_iou','ap','heldout_teacher_iou']},queries=query_rows)
    baseline={(q['scene'],q['query']):q['iou'] for q in arms['fit']['queries']}
    for arm in ['conditioned','forced_split']:
        arms[arm]['changes']=sorted([dict(q,delta=q['iou']-baseline[q['scene'],q['query']]) for q in arms[arm]['queries']],key=lambda r:r['delta'])
    full_baseline={(q['scene'],q['query']):q['iou'] for q in arms['full_reference']['queries']}
    arms['full_conditioned']['changes']=sorted([dict(q,delta=q['iou']-full_baseline[q['scene'],q['query']]) for q in arms['full_conditioned']['queries']],key=lambda r:r['delta'])
    (root/'summary.json').write_text(json.dumps(dict(arms=arms,protocol='frozen development labels; source heldout is teacher consistency, not independent GT; no unseen-scene claims'),indent=2)+'\n')
    for arm,r in arms.items():print(arm,r['macro'])


if __name__=='__main__':main()
