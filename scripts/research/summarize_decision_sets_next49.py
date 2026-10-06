"""Summarize fixed-protocol development controls and carrier pilots."""
import json
from pathlib import Path
import numpy as np
OUT=Path('output/decision_sets_next49')
scenes=['figurines','ramen','teatime','waldo_kitchen']
controls={}
for arm in ['baseline','decision','bernoulli','both','proposal_unknown','proposal_positive','enclosure_required','proposal_decision']:
    rows={};focus={};admission={}
    for scene in scenes:
        base=OUT/arm/('lerf_'+scene);p=base/'anchored/pipeline_calibrated/evaluation.json'
        if not p.exists():continue
        m=json.loads(p.read_text())['metrics']['hierarchical_pipeline'];rows[scene]=m
        fit=json.loads((base/'fit.json').read_text())
        admission[scene]=dict(abstained=[r['query'] for r in fit['history'] if r.get('abstained')],enclosure_observations=len(fit.get('proposal_relations',[])),seconds=fit['seconds'])
        for query in ['coffee','refrigerator']:
            v=[r['iou'] for r in m['selected']['objects'] if r['query']==query]
            if v:focus[query]=float(np.mean(v))
    if rows:controls[arm]=dict(scenes=rows,macro_selected=float(np.mean([d['selected']['miou'] for d in rows.values()])),focus_queries=focus,source_admission=admission)
carrier={}
for arm in ['split_prior','split_fit']:
    rows={}
    for scene in ['figurines','ramen']:
        p=Path('output/refined_carrier_next49')/arm/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
        if p.exists():rows[scene]=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
    if rows:carrier[arm]=dict(scenes=rows,macro_selected=float(np.mean([d['selected']['miou'] for d in rows.values()])),selection_primitives_multiplier=4,equal_capacity=False)
typed_rows={}
for scene in scenes:
    p=OUT/'typed_enclosure'/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
    if p.exists():typed_rows[scene]=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
typed=dict(scenes=typed_rows,macro={k:float(np.mean([m[k] for m in typed_rows.values()])) for k in ['iou','ap','localization']},macro_selected=float(np.mean([m['selected']['miou'] for m in typed_rows.values()]))) if typed_rows else {}
summary=dict(typed_enclosure=typed,development_only=True,controls=controls,carrier_pilots=carrier,mainline_default_changed=False,no_new_scannet_nvos_uqis_results=True)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({a:dict(macro_selected=r['macro_selected'],scenes={s:v['selected']['miou'] for s,v in r['scenes'].items()},focus=r['focus_queries']) for a,r in controls.items()},indent=2));print('carrier',{a:r['macro_selected'] for a,r in carrier.items()})
