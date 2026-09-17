import json
import pytest
from scripts.audits.aggregate_clean_nvos import aggregate, TASKS, RUNS
from radio_gs.utils.immutable_artifacts import sha256_file


def fixture_tree(tmp_path):
    for i, task in enumerate(TASKS):
        root=tmp_path/('horns_center' if task=='horns_left' else task)
        root.mkdir(exist_ok=True)
        (root/'execution.json').write_text('{"finished":true}')
        hashes={}
        for run in RUNS:
            f=root/'field'/run/'field.pt';f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'field')
            hashes[run]=sha256_file(f)
        for mode in ('image','signed'):
            out=root/'evaluation'/(task+'_'+mode);out.mkdir(parents=True)
            results={}
            for j, run in enumerate(RUNS):
                p=out/(run+'.pt');p.write_bytes(b'prediction')
                results[run]=dict(iou=(i+j)/20,average_precision=.5,pixel_accuracy=.5,positive_support_recall=1.,reference_signed_agreement={'positive_agreement':.9 if i==0 else 1.,'negative_agreement':1.},prediction_sha256=sha256_file(p))
            r=dict(scene=task,sam_reference_complex_rope_preserved=True,target_rgb_opened=False,target_gt_used_after_prediction_only=True,signed_refinement=mode=='signed',signed_policy='adaptive',signed_bound_control=None,threshold=.5,threshold_comparison='greater_or_equal',score_semantics='membership',score_resize='nearest',fields=hashes,results=results)
            (out/'report.json').write_text(json.dumps(r))
    return tmp_path


def test_fixed_all_task_macro_and_unmet_constraints(tmp_path):
    r=aggregate(fixture_tree(tmp_path))
    assert len(r['per_task'])==8
    for j, run in enumerate(RUNS):
        assert r['metrics']['signed'][run]['iou']==pytest.approx((3.5+j)/20)
        assert r['metrics']['signed'][run]['reference_constraints_not_fully_satisfied']==['fern']


def test_missing_scene_is_not_silently_excluded(tmp_path):
    fixture_tree(tmp_path)
    (tmp_path/'trex/execution.json').write_text('{"finished":false}')
    with pytest.raises(RuntimeError,match='trex'):aggregate(tmp_path)


def test_changed_prediction_is_rejected(tmp_path):
    fixture_tree(tmp_path)
    (tmp_path/'fern/evaluation/fern_signed'/f'{RUNS[0]}.pt').write_bytes(b'changed')
    with pytest.raises(AssertionError):aggregate(tmp_path)


def test_mixed_threshold_contract_is_rejected(tmp_path):
    fixture_tree(tmp_path)
    p=tmp_path/'leaves/evaluation/leaves_signed/report.json';r=json.loads(p.read_text());r['threshold']=.4;p.write_text(json.dumps(r))
    with pytest.raises(AssertionError):aggregate(tmp_path)
