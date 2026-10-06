"""Separate fixed dense-carrier pilots from the complete NVOS method cohort."""
import json
from pathlib import Path


def average(rows):return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}


def dense(base):
    result={}
    for mode in ['gaussian','gaussian_gate','surface','surface_robust']:
        rows={}
        for p in sorted(Path(base,mode).glob('*/anchored/pipeline/evaluation.json')):
            d=json.loads(p.read_text());m=d['metrics']['hierarchical_pipeline']
            rows[d['key']]=(dict(iou=m['iou'],ap=m['ap'],selected=m['selected']['miou']) if 'iou' in m else {s:m[s]['miou'] for s in ['19','15','10']})
        result[mode]=dict(scenes=rows,lerf_two_scene_mean=average([v for k,v in rows.items() if k.startswith('lerf')]))
    return result


def nvos(base):
    rows={}
    for p in sorted(Path(base).glob('nvos_*/evaluation_*/evaluation.json')):
        d=json.loads(p.read_text());m=d['metrics']['trained'];rows[d['scene']]=dict(iou=m['iou'],ap=m['average_precision'])
    return dict(scenes=rows,macro=average(list(rows.values())))


if __name__=='__main__':
    result=dict(development=True,independent_test=False,
                dense_raw_spatial_unaligned_diagnostic=dense('output/dense_surface_next43'),
                dense_radio_local_head=dense('output/dense_language_surface_matched_next43'),
                dense_native_local_head=dense('output/native_dense_surface_matched_next43'),
                nvos_parent=nvos('output/taskwise_scope_composition_next38'),
                nvos_reference_scopes=nvos('output/seeded_extent_cut_next40/reference_scopes'),
                nvos_positive_video=nvos('output/source_video_next43/positive'),
                nvos_signed_video=nvos('output/source_video_next43/signed'))
    result['source_readback']={base:{p.parent.name:json.loads(p.read_text())['macro'] for p in Path('output',base).glob('*/source_readback.json')} for base in ['dense_language_surface_next43','native_dense_surface_next43']}
    result['nvos_sources']={p.parent.name:json.loads(p.read_text()) for p in Path('output/source_video_next43').glob('*/tracking.json')}
    Path('output/dense_video_next43_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['macro'] for k,v in result.items() if k.startswith('nvos_') and 'macro' in v},indent=2))
    for k in ['dense_raw_spatial_unaligned_diagnostic','dense_radio_local_head','dense_native_local_head']:print(k,{m:v['lerf_two_scene_mean'] for m,v in result[k].items()})
