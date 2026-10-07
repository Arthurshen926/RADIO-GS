"""Precision/recall decomposition of already scored LERF subset rasters."""
import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask, calculate_iou
from .storage import sha256


def analyze(root, output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('preserve the existing diagnostic')
    reports={}
    for path in sorted(root.glob('*/evaluation/evaluation.json')):
        report=json.loads(path.read_text());controls={}
        for name,metric in report['metrics'].items():
            rows=[]
            for item in metric['selected']['objects']:
                if item['missing']:raise ValueError('missing prediction must be resolved before decomposition')
                gt=load_binary_mask(Path(item['gt_path']),threshold=10)
                prediction=load_binary_mask(Path(item['pred_path']),threshold=10,grayscale=True)
                value=calculate_iou(gt,prediction)
                if abs(value-item['iou'])>1e-12:raise ValueError('diagnostic differs from retained scorer')
                intersection=int((gt&prediction).sum())
                rows.append(dict(frame=item['frame'],query=item['query'],iou=value,
                    precision=intersection/max(1,int(prediction.sum())),recall=intersection/max(1,int(gt.sum())),
                    prediction_area=float(prediction.mean()),target_area=float(gt.mean())))
            controls[name]=dict(cases=len(rows),rows=rows,
                mean_precision=float(np.mean([r['precision'] for r in rows])),
                mean_recall=float(np.mean([r['recall'] for r in rows])),
                mean_iou=float(np.mean([r['iou'] for r in rows])))
        reports[path.parent.parent.name]=dict(evaluation_sha256=sha256(path),controls=controls)
    if not reports:raise ValueError('no completed evaluation reports')
    result=dict(schema='radio_gs.query_field_subset_errors.v2',threshold=10,
        ground_truth_and_prediction_use_retained_loader=True,every_iou_replayed=True,reports=reports)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return {name:{key:{k:v for k,v in value.items() if k!='rows'} for key,value in report['controls'].items()}
            for name,report in reports.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.root,args.output),indent=2))


if __name__=='__main__':main()
