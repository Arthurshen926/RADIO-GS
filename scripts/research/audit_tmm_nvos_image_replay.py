"""Compare complete N04 prediction banks before release; no target labels used."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.replay_tmm_response import STUDY,POLICY,sha
from scripts.research.run_tmm_support_agreement import array_sha


def main():
    roots=[STUDY/r for r in ['nvos_image_response_v2','nvos_image_response_v3']]
    manifests=[json.loads((r/'manifest.json').read_text()) for r in roots]
    seals=[json.loads((r/'predictions_complete.json').read_text()) for r in roots]
    assert all(s['images']==56 for s in seals)
    banks=[Path(m['volatile_prediction_directory']) for m in manifests]
    entries=[json.loads((b/'manifest.json').read_text()) for b in banks]
    rows={}
    for job in manifests[0]['jobs']:
        name=job['name'];spec=json.loads(Path(job['evaluation']).read_text())
        # Only the image dimensions are needed, not a foreground/background label.
        with Image.open(spec['gt_mask']) as image:wh=image.size
        for arm in manifests[0]['config']['arms']:
            records=[e[str(r)+'::'+name+'_'+arm] for r,e in zip(roots,entries)]
            values=[np.load(b/e['file'],allow_pickle=False) for b,e in zip(banks,records)]
            assert all(array_sha(x)==s['predictions'][name][arm]['sha256'] for x,s in zip(values,seals))
            x,y=values
            a,b=[cv2.resize(v.reshape(spec['raster_wh'][1],spec['raster_wh'][0]),wh,interpolation=cv2.INTER_LINEAR) for v in values]
            rows[name+'_'+arm]=dict(exact_float_hash=records[0]['sha256']==records[1]['sha256'],
                maximum_float_error=float(np.abs(x-y).max()),mean_float_error=float(np.abs(x-y).mean()),
                native_binary_changed=int(((x>=.5)!=(y>=.5)).sum()),scoring_grid_binary_changed=int(((a>=.5)!=(b>=.5)).sum()))
    run=BoundedExperiment(STUDY,'nvos_image_response_v3',POLICY,resume=True)
    run.json('replay_comparison.json',dict(rows=rows,images=56,target_labels_used=False,
        target_metadata_used='GT raster dimensions only',source_hash=sha(__file__),source_text=Path(__file__).read_text(),
        prior_seal_sha256=sha(roots[0]/'predictions_complete.json'),canonical_seal_sha256=sha(roots[1]/'predictions_complete.json'),
        exact_float_images=sum(v['exact_float_hash'] for v in rows.values()),
        max_float_error=max(v['maximum_float_error'] for v in rows.values()),
        native_binary_changed=sum(v['native_binary_changed'] for v in rows.values()),
        scoring_grid_binary_changed=sum(v['scoring_grid_binary_changed'] for v in rows.values())))
    print(json.dumps({k:v for k,v in json.loads((run.root/'replay_comparison.json').read_text()).items() if k not in ('rows','source_text')},indent=2))


if __name__=='__main__':main()
