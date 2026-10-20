"""Publication regressions: wrong attribution, cohort mixing, metric substitution."""
import copy
import json
from pathlib import Path

import pytest

from scripts.research.build_tmm_paper_experiments import checked_mean, validate_row

CONFIG = json.loads((Path(__file__).resolve().parents[1]/
    'configs/research/tmm_paper_experiments.json').read_text())


def test_equal_scene_macro_is_not_an_object_pooled_mean():
    # One perfect scene with one query and a failed scene with 100 queries
    # still have equal scene weight under the retained LERF protocol.
    rows={'small':dict(miou=1.,queries=1),'large':dict(miou=0.,queries=100)}
    assert checked_mean(rows,['small','large'],'miou')==.5


@pytest.mark.parametrize('actual', [list(range(7)),list(range(9)),list(range(10)),list(range(1,9))])
def test_paper8_rejects_missing_extra_or_substituted_rooms(actual):
    with pytest.raises(ValueError,match='cohort mismatch'):
        checked_mean({s:dict(miou=.5) for s in actual},list(range(8)),'miou')


def test_missing_method_result_cannot_be_filled_from_frozen_provider():
    row=dict(benchmark='lerf2d',kind='pending',metrics_percent=dict(miou=None,localization_accuracy=None))
    validate_row(row,CONFIG)
    row['metrics_percent']['miou']=70.4
    with pytest.raises(ValueError,match='pending mechanism'):
        validate_row(row,CONFIG)


def test_ap_cannot_silently_replace_localization_accuracy():
    row=dict(benchmark='lerf2d',kind='provider',metrics_percent=dict(miou=70.4,ap=87.6))
    with pytest.raises(ValueError,match='metric identity'):
        validate_row(row,CONFIG)


def test_primary_source_corrects_historical_vala_drsplat_misattribution():
    rows={r['method']:r for r in CONFIG['published_rows'] if r['benchmark']=='lerf3d'}
    assert rows['VALA']['metrics_percent']==dict(miou=58.02,accuracy_at_iou_0p25=82.85)
    assert rows['Dr. Splat']['metrics_percent']==dict(miou=43.29,accuracy_at_iou_0p25=64.30)
    assert rows['VALA']['source']=='vala_lerf'
    assert CONFIG['sources']['vala_lerf']['table']=='1'
    assert not any(r['benchmark']=='uqis' for r in CONFIG['published_rows'])


def test_external_row_requires_real_reporting_source():
    row=copy.deepcopy(CONFIG['published_rows'][0]);row.pop('source')
    with pytest.raises(ValueError,match='reporting source'):
        validate_row(row,CONFIG)
