import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_query_preserving_field import example
from radio_gs.query_field.storage import save_field
from radio_gs.query_field.public_lerf import predict,score,selection_scores


def test_prediction_needs_no_evaluation_scene_and_owns_one_subset(tmp_path,monkeypatch):
    field=example();save_field(field,tmp_path/'field',{})
    queries=tmp_path/'queries.json'
    queries.write_text(json.dumps({'queries':[{'modality':'text','text':'target'}]}))
    class Encoder:
        def __init__(self,*args):pass
        def text(self,text):
            return SimpleNamespace(semantic=field.semantic.codes[0 if text=='target' else 1])
    monkeypatch.setattr('radio_gs.query_field.public_lerf.NativeQueryEncoder',Encoder)
    config={'shared_encoder':'not_required_in_mock','generic_negative_texts':['object'],
            'native_temperature':10.,'jobs':[]}
    job={'name':'one','field':str(tmp_path/'field'),'queries':str(queries),'evaluation_scene':'/not/available/scene.json'}
    config['jobs']=[job]
    report=predict(config,job,tmp_path/'result',device='cpu')
    assert not report['runtime_source_access'] and not report['legacy_calibration_reused']
    payload=torch.load(tmp_path/'result/one/predictions.pt',weights_only=True)
    assert set(payload['scores'])=={'native_generic_relevance','shared_membership'}
    assert torch.equal(payload['selected'],payload['scores']['shared_membership']>.5)
    assert payload['scores']['native_generic_relevance'].shape==payload['selected'].shape
    from radio_gs.query_field.readout_study import predict as predict_controls
    control_job=dict(job,reference_prediction=str(tmp_path/'result/one'))
    controls_report=predict_controls({'jobs':[control_job]},control_job,tmp_path/'controls',device='cpu')
    assert controls_report['union_replayed_bit_exact']
    controls=torch.load(tmp_path/'controls/one/predictions.pt',weights_only=True)
    assert torch.equal(controls['selected_by_control']['union_member'],payload['selected'])
    assert torch.equal(controls['selected_by_control']['native_only'],
                       (payload['scores']['native_generic_relevance']>.5)&payload['observed'][:,None])
    config['jobs'].append(dict(job,name='missing'))
    with pytest.raises(ValueError,match='changed after predictions'):
        score(config,job,tmp_path/'result')


def test_scoring_cannot_open_labels_before_all_predictions_exist(tmp_path):
    config={'jobs':[{'name':'not_predicted'}]}
    with pytest.raises(ValueError,match='complete all fixed predictions'):
        score(config,config['jobs'][0],tmp_path)


def test_readout_controls_keep_their_own_selection():
    scores={'a':torch.zeros(3,2),'b':torch.ones(3,2)}
    sets={key:value.bool() for key,value in scores.items()}
    values=selection_scores(dict(scores=scores,selected_by_control=sets),'cpu')
    assert values['a'].sum()==0 and values['b'].sum()==6
    with pytest.raises(ValueError,match='declared score controls'):
        selection_scores(dict(scores=scores,selected_by_control={'a':sets['a']}),'cpu')


def test_subset_decomposition_replays_the_retained_ground_truth_threshold(tmp_path):
    import numpy as np
    from PIL import Image
    from radio_gs.query_field.subset_errors import analyze
    gt=tmp_path/'gt.png';prediction=tmp_path/'prediction.png'
    # The retained loader counts the antialias value 20 as positive, too.
    Image.fromarray(np.array([[0,20,255]],dtype=np.uint8)).save(gt)
    Image.fromarray(np.array([[0,20,0]],dtype=np.uint8)).save(prediction)
    directory=tmp_path/'job/evaluation';directory.mkdir(parents=True)
    obj=dict(frame='one',query='target',gt_path=str(gt),pred_path=str(prediction),missing=False,iou=.5)
    (directory/'evaluation.json').write_text(json.dumps({'metrics':{'control':{'selected':{'objects':[obj]}}}}))
    result=analyze(tmp_path,tmp_path/'errors.json')['job']['control']
    assert result['mean_precision']==1 and result['mean_recall']==.5 and result['mean_iou']==.5
