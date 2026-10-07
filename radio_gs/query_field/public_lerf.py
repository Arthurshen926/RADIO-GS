"""Two explicit semantic controls with one shared actual Gaussian subset.

Prediction consumes the exported field and current text only. The separate
scoring stage reuses the unchanged historical LERF scorer and its cameras.
"""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .adapters import NativeQueryEncoder
from .contracts import QueryPacket
from .storage import load_field,sha256


def selection_scores(data, device):
    """Keep each diagnostic set paired with its explicitly named readout."""
    controls = data.get('selected_by_control')
    if controls is None:
        controls = {key:data['selected'] for key in data['scores']}
    if set(controls) != set(data['scores']):
        raise ValueError('selection controls differ from declared score controls')
    if any(value.shape != data['scores'][key].shape for key,value in controls.items()):
        raise ValueError('selection and response rows differ')
    return {key:value.float().to(device) for key,value in controls.items()}


@torch.no_grad()
def predict(config,job,output,device='cuda'):
    out=Path(output)/job['name'];out.mkdir(parents=True,exist_ok=False)
    queries=json.loads(Path(job['queries']).read_text())['queries']
    if any(q['modality']!='text' for q in queries):raise ValueError('probe requires text queries')
    texts=[q['text'] for q in queries]
    if len(texts)!=len(set(texts)):raise ValueError('duplicate query strings')
    encoder=NativeQueryEncoder(config['shared_encoder'],device)
    vectors=torch.stack([encoder.text(text).semantic.cpu().float() for text in texts])
    negatives=torch.stack([encoder.text(text).semantic.cpu().float() for text in config['generic_negative_texts']])
    del encoder;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    field,manifest=load_field(job['field'],device)
    background=field.semantic.score(negatives).max(1).values
    native=[];members=[];selection=[];diagnostics=[]
    for text,vector in zip(texts,vectors):
        result=field.query(QueryPacket('text',task='union',semantic=vector.to(device)))
        native.append(torch.sigmoid(config['native_temperature']*(result.semantic_scores-background)).cpu())
        members.append(result.membership_scores.cpu())
        selection.append(result.selected.cpu())
        diagnostics.append(dict(text=text,selected=int(result.selected.sum()),**result.diagnostics))
        print('predicted',job['name'],text,flush=True)
    geometry={key:getattr(field,key).cpu() for key in ['xyz','normals','quaternions','scales','opacities']}
    payload=dict(texts=texts,geometry=geometry,observed=field.observed.cpu(),
        scores=dict(native_generic_relevance=torch.stack(native,1),shared_membership=torch.stack(members,1)),
        selected=torch.stack(selection,1),native_query_vectors=vectors,generic_negative_vectors=negatives)
    torch.save(payload,out/'predictions.pt')
    report=dict(schema='radio_gs.query_field_lerf_prediction.v1',job=job,
        field_sha256=manifest['files']['field.pt']['sha256'],query_file_sha256=sha256(job['queries']),
        predictions_sha256=sha256(out/'predictions.pt'),queries=len(texts),runtime_source_access=False,
        legacy_calibration_reused=False,configuration=config,diagnostics=diagnostics,
        geometry_source='exported field only',evaluation_scene_opened_by_predictor=False)
    (out/'prediction.json').write_text(json.dumps(report,indent=2)+'\n')
    return {key:report[key] for key in ['job','queries','predictions_sha256','runtime_source_access','legacy_calibration_reused']}


@torch.no_grad()
def score(config,job,output,device='cuda'):
    if not str(device).startswith('cuda'):raise ValueError('retained LERF scorer requires CUDA')
    root=Path(output)
    # All fixed arms must have completed predictions before any annotation load.
    for other in config['jobs']:
        directory=root/other['name']
        if not (directory/'prediction.json').is_file():raise ValueError('complete all fixed predictions before scoring')
        record=json.loads((directory/'prediction.json').read_text())
        if record['configuration']!=config:raise ValueError('probe configuration changed after predictions')
    directory=root/job['name'];receipt=json.loads((directory/'prediction.json').read_text())
    path=directory/'predictions.pt'
    if sha256(path)!=receipt['predictions_sha256']:raise ValueError('predictions changed before scoring')
    out=directory/'evaluation';out.mkdir(exist_ok=False)
    data=torch.load(path,map_location='cpu',weights_only=True)
    scene=json.loads(Path(job['evaluation_scene']).read_text())
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.evaluation.lerf_readouts import evaluate_lerf_readouts
    # Confirm exact geometry equality in the scoring process. This original
    # geometry asset and cameras were not accessed by the method predictor.
    original=FrozenGaussianGeometry.load(scene['geometry']['path'],expected_sha256=scene['geometry']['sha256'],device='cpu')
    for key,value in data['geometry'].items():
        if not torch.equal(value,getattr(original,key)):raise ValueError('retained evaluation carrier differs')
    geometry=SimpleNamespace(**{key:value.to(device) for key,value in data['geometry'].items()})
    observed=data['observed'].to(device)
    scores={key:value.to(device) for key,value in data['scores'].items()}
    sets=selection_scores(data,device)
    result=evaluate_lerf_readouts(scene=job['scene'],out=out,texts=data['texts'],scores=scores,selection_scores=sets,
        geo=geometry,default_observed=observed,semantic_geo=geometry,semantic_observed=observed,
        selection_geo=geometry,selection_observed=observed,camera_report=scene['camera_report'],
        camera_report_sha256=scene['camera_report_sha256'])
    report=dict(schema='radio_gs.query_field_lerf_evaluation.v1',job=job,metrics=result,
        prediction_sha256=receipt['predictions_sha256'],unchanged_retained_scorer=True,
        evaluation_scene_sha256=sha256(job['evaluation_scene']),benchmark_result=True,independent_final_test=False,
        source_budget_different_from_v10=True,legacy_calibration_reused=False,
        selected_projection_is_not_3d_ground_truth=True,promotion=False)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['predict','score']);parser.add_argument('--config',required=True)
    parser.add_argument('--job',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args();torch.set_num_threads(4)
    config=json.loads(Path(args.config).read_text());job=next(j for j in config['jobs'] if j['name']==args.job)
    function=predict if args.stage=='predict' else score
    print(json.dumps(function(config,job,args.output,args.device),indent=2))


if __name__=='__main__':main()
