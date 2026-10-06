"""Aggregate complete cohorts separately from mechanism pilots."""
import json
from pathlib import Path
OUT=Path('output/typed_evidence_next44')


def mean(rows):return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]} if rows else {}

def text_results(base):
    scenes={}
    for p in Path(base).glob('*/anchored/pipeline_calibrated/evaluation.json'):
        d=json.loads(p.read_text());m=d['metrics']['hierarchical_pipeline']
        scenes[d['key']]=({k:m[k] for k in ['iou','ap','localization']}|{'selected':m['selected']['miou']}) if 'iou'in m else {k:m[k]['miou'] for k in ['19','15','10']}
    lerf=[v for k,v in scenes.items() if k.startswith('lerf_')];scan=[v for k,v in scenes.items() if k.startswith('scannet_')]
    return dict(scenes=scenes,lerf_count=len(lerf),scannet_count=len(scan),lerf_macro=mean(lerf),scannet_macro=mean(scan))

def nvos_results(base):
    scenes={}
    for p in Path(base).glob('nvos_*/evaluation_*/evaluation.json'):
        d=json.loads(p.read_text());m=d['metrics']['trained'];scenes[d['scene']]=dict(iou=m['iou'],ap=m['average_precision'])
    return dict(scenes=scenes,count=len(scenes),macro=mean(list(scenes.values())))

if __name__=='__main__':
    report=dict(development=True,independent_test=False,parent=text_results('output/taskwise_scope_composition_next38'),text={m:text_results(OUT/m) for m in ['direct','agreement','corroborated','identity','identity_consensus','soft_identity']},
                nvos_parent=nvos_results('output/taskwise_scope_composition_next38'),nvos_next43=nvos_results('output/source_video_next43/signed'),nvos={m:nvos_results(OUT/m) for m in ['agreement','corroborated','consensus','bridge','full_video']},
                extra_online_source_rgb_and_vfm_budget=True,target_rgb_used=False,geometry_frozen=True,original_map_features_frozen=True)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for m,r in report['text'].items():print(m,r['lerf_count'],r['scannet_count'],r['lerf_macro'],r['scannet_macro'])
    for m,r in report['nvos'].items():print('NVOS',m,r['count'],r['macro'])
