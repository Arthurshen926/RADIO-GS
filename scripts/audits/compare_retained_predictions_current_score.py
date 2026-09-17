"""Rescore sealed historical binary predictions on the current 480-wide GT grid.

This aligns scoring, not source information, threshold selection or rendering.
Historical predictions are nearest-neighbor resampled, not rerendered.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks,binary_iou
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    output=Path('paper/artifacts/historical_current_score_alignment_20260916.json')
    if output.exists():raise FileExistsError(output)
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output/v4_repair_control_20260914')
    result=dict(schema='historical_current_binary_score_alignment.v1',scenes={},
        score_grid_aligned=True,full_method_protocol_aligned=False,independent_final_test=False,
        historical_prediction_resampling='PIL nearest to current 480-wide raster; no rethreshold or rerender',
        caveat='legacy selected-only output, source/query access and historical calibration remain different; not a controlled architecture comparison',
        implementation_sha256=sha256_file(__file__))
    for scene in ['figurines','ramen']:
        parent=base/scene;receipt=json.loads((parent/'replay_receipt.json').read_text())
        assert sha256_file(receipt['source_result'])==receipt['source_sha256']
        current_path=Path(f'paper/artifacts/{scene}_source_extent_residual_text_20260916.json')
        current=json.loads(current_path.read_text());run=current['runs']['source_extent_rendered_residual96']
        for label in current['protocol']['labels']:assert sha256_file(label['path'])==label['sha256']
        ann,categories,sh,sw=load_labels(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'),scene)
        width=480;height=round(sh*width/sw);targets={f:ground_truth_masks(o,categories,height,width,sh,sw) for f,o in ann.items()}
        rows=[]
        for observation in run['observations']:
            frame=observation['frame'];category=observation['category']
            path=parent/'pred_masks/thr0p6'/scene/f'frame_{frame:05d}_{category}.png'
            im=Image.open(path);array=np.asarray(im);assert array.ndim==2 and set(np.unique(array)).issubset({0,255})
            pred=torch.from_numpy(np.asarray(im.resize((width,height),Image.Resampling.NEAREST)).copy())>127
            iou,intersection,union=binary_iou(pred,targets[frame][category])
            rows.append(dict(frame=frame,category=category,historical_iou=iou,current_iou=observation['iou'],
                intersection=intersection,union=union,prediction=str(path),prediction_sha256=sha256_file(path),original_size=list(im.size)))
        assert len(list((parent/'pred_masks/thr0p6'/scene).glob('*.png')))==len(rows)
        result['scenes'][scene]=dict(observations=rows,tasks=len(rows),
            historical_observation_miou=sum(r['historical_iou'] for r in rows)/len(rows),current_observation_miou=run['observation_miou'],
            historical_category_macro=sum(sum(r['historical_iou'] for r in rows if r['category']==c)/sum(r['category']==c for r in rows) for c in categories)/len(categories),
            historical_receipt_sha256=sha256_file(parent/'replay_receipt.json'),current_result_sha256=sha256_file(current_path))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print({s:(v['historical_observation_miou'],v['current_observation_miou']) for s,v in result['scenes'].items()})


if __name__=='__main__':main()
