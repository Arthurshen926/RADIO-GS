"""Attribute frozen predictions after evaluation; no policy fitting or selection."""
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS,NYU40_ID_TO_NAME

ROOT=Path('output/element_retrieval_reliability')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--baseline',default='adapter_vote')
    args=parser.parse_args();root=args.root
    summary=json.loads((root/'summary.json').read_text());rows={}
    for name in json.loads((root/'split.json').read_text())['validation']:
        spec=json.loads((root/f'{name}.evaluation.json').read_text())
        truth=np.asarray(PlyData.read(spec['ground_truth'])['vertex']['label']);valid=np.isin(truth,SPLITS['19'])
        path=root/'evaluation'/args.baseline/name/'evaluation'
        baseline=np.load(path/'source_point_predictions.npz')['19']
        rows[name]={}
        for mode in summary['scenes']:
            pred=np.load(root/'evaluation'/mode/name/'evaluation/source_point_predictions.npz')['19']
            prior=np.load(root/'evaluation'/mode/name/'evaluation/map_point_predictions.npz')['19']
            point=dict(points=int(valid.sum()),native_correct_to_wrong=int((valid&(prior==truth)&(pred!=truth)).sum()),
                native_wrong_to_correct=int((valid&(prior!=truth)&(pred==truth)).sum()),
                vote_correct_to_wrong=int((valid&(baseline==truth)&(pred!=truth)).sum()),
                vote_wrong_to_correct=int((valid&(baseline!=truth)&(pred==truth)).sum()))
            classes={}
            for c in SPLITS['19']:
                gt=truth==c
                if not gt.any():continue
                selected=(pred==c)&valid;reference=(baseline==c)&valid
                classes[NYU40_ID_TO_NAME[c]]=dict(iou=float((selected & gt).sum()/(selected | gt).sum()),
                    vote_iou=float((reference & gt).sum()/(reference | gt).sum()),
                    false_positive=int((selected & ~gt).sum()),false_negative=int((~selected & gt).sum()))
            rows[name][mode]=dict(**point,classes=classes)
    (root/'outcomes.json').write_text(json.dumps(dict(scenes=rows,baseline=args.baseline,post_prediction_only=True,labels_used_only_for_measurement=True),indent=2)+'\n')
    print({name:{mode:(d['vote_wrong_to_correct'],d['vote_correct_to_wrong']) for mode,d in modes.items()} for name,modes in rows.items()})

if __name__=='__main__':main()
