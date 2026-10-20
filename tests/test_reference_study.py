import pytest

from radio_gs.response_study.reference_study import check_binary_metrics


def test_confusion_counts_bind_all_reported_metrics():
    row=dict(tp=6,fp=2,fn=4,iou=.5,precision=.75,recall=.6)
    check_binary_metrics(row)
    row['iou']=.75
    with pytest.raises(ValueError,match='counts and metrics'):check_binary_metrics(row)


def test_empty_prediction_and_truth_use_explicit_conventions():
    check_binary_metrics(dict(tp=0,fp=0,fn=0,iou=1.,precision=0.,recall=0.))
    check_binary_metrics(dict(tp=0,fp=0,fn=5,iou=0.,precision=0.,recall=0.))


def test_nonfinite_score_or_negative_count_is_rejected():
    with pytest.raises(ValueError):
        check_binary_metrics(dict(tp=1,fp=0,fn=0,iou=float('nan'),precision=1.,recall=1.))
    with pytest.raises(ValueError):
        check_binary_metrics(dict(tp=1,fp=-1,fn=0,iou=1.,precision=1.,recall=1.))
