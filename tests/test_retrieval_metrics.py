import torch
from radio_gs.training.retrieval_metrics import partial_retrieval_metrics


def test_unknown_predictions_are_reported_but_not_counted_as_false_positive():
    p=torch.tensor([.9,.1,.99,.99]);target=torch.tensor([1,0,0,0]);known=torch.tensor([1,1,0,0],dtype=torch.bool)
    support=torch.tensor([1,1,1,0],dtype=torch.bool)
    m=partial_retrieval_metrics(p,target,known,support)
    assert m['known_iou']==1
    assert m['known_negative_false_positive_rate']==0
    assert m['unknown_positive_pixel_fraction']==.25
    assert .49 < m['unknown_response_mass_fraction'] < .50


def test_empty_labels_are_undefined_not_perfect_or_zero_quality():
    m=partial_retrieval_metrics(torch.tensor([.9]),torch.tensor([0]),torch.tensor([False]))
    assert m['known_iou'] is None
    assert m['known_positive_recall'] is None
    assert m['known_negative_false_positive_rate'] is None


def test_partial_macro_exposes_undefined_episode_denominator():
    from radio_gs.training.retrieval_metrics import summarize_partial_episodes
    result = summarize_partial_episodes([
        {'known_iou': None, 'known_precision': None, 'unknown_is_negative': False},
        {'known_iou': .8, 'known_precision': 1., 'unknown_is_negative': False},
    ])
    assert result['known_iou'] == .8
    assert result['known_iou_all_episodes_zero_for_undefined'] == .4
    assert result['episode_count'] == 2
    assert result['defined_episode_counts']['known_iou'] == 1


def test_coverage_aware_metrics_count_uncovered_positive_as_miss():
    from radio_gs.training.retrieval_metrics import coverage_aware_retrieval_metrics
    result = coverage_aware_retrieval_metrics(
        torch.tensor([.9, .99, .99, .99]), torch.tensor([1, 1, 0, 0]),
        torch.tensor([True, True, True, False]),
        torch.tensor([True, False, False, True]),
    )
    assert result['known_iou'] == .5
    assert result['known_positive_recall'] == .5
    assert result['label_positive_support_recall'] == .5
    assert result['unsupported_known_positive_count'] == 1
    assert result['known_negative_false_positive_rate'] == 0


def test_coverage_aware_empty_map_is_not_perfect_on_positive_labels():
    from radio_gs.training.retrieval_metrics import coverage_aware_retrieval_metrics
    result = coverage_aware_retrieval_metrics(
        torch.tensor([.9]), torch.tensor([1]), torch.tensor([True]), torch.tensor([False]))
    assert result['known_iou'] == 0
    assert result['known_positive_recall'] == 0


def test_ranking_metrics_perfect_reversed_and_ties():
    from radio_gs.training.retrieval_metrics import binary_ranking_metrics
    target=torch.tensor([1,0])
    assert binary_ranking_metrics(torch.tensor([.9,.1]),target)['average_precision']==1
    assert binary_ranking_metrics(torch.tensor([.9,.1]),target)['auroc']==1
    assert binary_ranking_metrics(torch.tensor([.1,.9]),target)['auroc']==0
    tied=binary_ranking_metrics(torch.tensor([.5,.5]),target)
    assert tied['average_precision']==.5
    assert tied['auroc']==.5
    assert binary_ranking_metrics(torch.tensor([.5,.5]),target.flip(0))==tied
