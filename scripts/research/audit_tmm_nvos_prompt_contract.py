"""Check all original positive scribbles against retained source seeds, without GT."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.replay_tmm_response import STUDY,POLICY,sha


def main():
    recipe=Path('configs/pipeline/system_v10.json')
    jobs=[j for j in json.loads(recipe.read_text())['jobs'] if j['benchmark']=='nvos']
    assert len(jobs)==8
    rows={};inputs={str(recipe):sha(recipe)}
    for j in jobs:
        q=json.loads(Path(j['queries']).read_text())
        assert not q['reference_mask_is_gt'] and not q['negative_prompt_used']
        scribble=np.asarray(Image.open(q['positive_scribble']))
        original_shape=list(scribble.shape)
        if scribble.ndim==3:
            assert scribble.shape[2]==3 and np.array_equal(scribble[...,0],scribble[...,1]) and np.array_equal(scribble[...,0],scribble[...,2])
            scribble=scribble[...,0]
        assert scribble.ndim==2 and set(np.unique(scribble).tolist())=={0,255}
        seed=np.asarray(Image.open(q['reference_mask']))
        assert seed.shape==scribble.shape and list(seed.shape[::-1])==q['prompt_raster_wh']
        positive=scribble>0;seed=seed>127
        count,labels,stats,_=cv2.connectedComponentsWithStats(positive.astype(np.uint8),connectivity=8)
        covered=int((positive&seed).sum());total=int(positive.sum())
        rows[j['name']]=dict(scribble_shape=original_shape,positive_pixels=total,covered_pixels=covered,
            missed_pixels=total-covered,positive_recall=covered/total,
            components=[dict(pixels=int(stats[i,cv2.CC_STAT_AREA]),covered_pixels=int(seed[labels==i].sum())) for i in range(1,count)],
            target_gt_read=False,source_seed_is_gt=False)
        for p in [j['queries'],q['positive_scribble'],q['reference_mask']]:inputs[str(p)]=sha(p)
    run=BoundedExperiment(STUDY,'nvos_unboxed_image_response_v1',POLICY,resume=True)
    run.json('source_prompt_contract.json',dict(complete=True,jobs=8,rows=rows,input_hashes=inputs,
        target_gt_read=False,unchanged_inference=True,source_text=Path(__file__).read_text(),source_sha256=sha(__file__)))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
