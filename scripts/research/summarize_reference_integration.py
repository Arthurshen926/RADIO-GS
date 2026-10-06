"""Keep frozen NVOS metrics and selected-projection diagnostics separate."""
import json
from pathlib import Path


def main():
    root=Path('output/reference_integration');results={}
    for arm in ['replay','fresh','subset','typed']:
        reports=[json.loads(p.read_text()) for p in sorted((root/arm).glob('*/evaluation/evaluation.json'))]
        results[arm]=dict(count=len(reports),scenes=[dict(scene=r['scene'],**r['metrics']) for r in reports],macro={k:sum(r['metrics'][k] for r in reports)/len(reports) for k in ['iou','average_precision']})
    old={s:json.loads(Path(f'output/nvos_source_resolution_next74/new/{s}/report.json').read_text())['metrics'] for s in [r['scene'] for r in results['fresh']['scenes']]}
    results['fresh_minus_retained']={r['scene']:r['iou']-old[r['scene']]['iou'] for r in results['fresh']['scenes']}
    results['subset_projection_diagnostic']=json.loads((root/'subset_projection_diagnostic.json').read_text())
    results['query_contract']='existing positive-only reference SAM mask; fresh tracking and reconstruction; no signed prompt/image-only/3D point claims'
    results['readout_contract']='NVOS continuous response>=0.5; selected silhouette reported only as separate diagnostic'
    (root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for arm in ['replay','fresh','subset','typed']:print(arm,results[arm]['count'],results[arm]['macro'])


if __name__=='__main__':main()
