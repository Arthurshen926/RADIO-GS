"""Summarize the completed mechanism experiments, keeping oracle rows separate."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
OUT=Path('output/vala_exchange_next48')
scenes=['figurines','ramen','teatime','waldo_kitchen'];result={}
for arm in ['rgb_all','rgb20','moge20','moge_fit20']:
    scene_rows={}
    for scene in scenes:
        base=OUT/arm/scene
        if not (base/'none/predictions_mask_0.6/result.txt').exists():continue
        rows=[]
        for gt in sorted((base/'gt').glob('*/*.jpg')):
            pred=base/'none/predictions_mask_0.6/renders_silhouette'/gt.parent.name/(gt.stem+'.png')
            g=np.asarray(Image.open(gt))>10;p=np.asarray(Image.open(pred).convert('L'))>10
            iou=float(np.logical_and(g,p).sum()/max(1,np.logical_or(g,p).sum()))
            rows.append(dict(frame=gt.parent.name,query=gt.stem,iou=iou))
        values=[r['iou'] for r in rows];scene_rows[scene]=dict(iou=float(np.mean(values)),acc05=float(np.mean(np.array(values)>.5)),observations=len(rows),rows=rows)
    result[arm]=dict(scenes=scene_rows,macro_iou=float(np.mean([r['iou'] for r in scene_rows.values()])))
oracle={}
for p in Path('output/carrier_oracle_next48').glob('*/*/diagnostic.json'):
    d=json.loads(p.read_text());oracle[p.parent.parent.name+'/'+p.parent.name]=dict(continuous_iou=d['continuous_iou'],selected_iou=d['selected']['miou'],gt_assisted_fit_and_score_on_same_views=True)
decoder={}
for scene in scenes:
    p=Path('output/set_decoder_next48/set')/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
    if p.exists():decoder[scene]=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
controls={}
for arm in ['set60','set','joint','semantic_unknown','field_unknown','field_agreement','typed','typed_field','typed_agreement']:
    cohort={}
    for scene in scenes:
        p=Path('output/set_decoder_next48')/arm/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json'
        if p.exists():cohort[scene]=json.loads(p.read_text())['metrics']['hierarchical_pipeline']
    if cohort:controls[arm]=dict(scenes=cohort,macro_iou=float(np.mean([d['iou'] for d in cohort.values()])),macro_selected=float(np.mean([d['selected']['miou'] for d in cohort.values()])))
summary=dict(official_code_geometry_exchange=result,gt_assisted_diagnostics_not_method_scores=oracle,source_only_set_decoder=decoder,decoder_controls=controls,retained_lerf_development_candidate='field_agreement',official_author_reproduction=False,equal_geometry_training_budget=False,mainline_promoted=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({a:{'iou':r['macro_iou'],'scenes':{s:v['iou'] for s,v in r['scenes'].items()}} for a,r in result.items()},indent=2));print('oracle',oracle);print('decoder',{s:(d['iou'],d['selected']['miou']) for s,d in decoder.items()})
