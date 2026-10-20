"""New studies must remain inspectable before their sealed outputs exist."""
import pytest
from radio_gs.response_study.target_redecode_study import check_target_redecode_study
from radio_gs.response_study.multiview_completion_study import check_multiview_completion_study
from radio_gs.response_study.reference_completion_study import check_reference_completion_study
from radio_gs.response_study.reference_delta_study import check_reference_delta_study
from radio_gs.response_study.reference_admission_study import check_reference_admission_study
from radio_gs.response_study.score_precision_audit import check_score_precision_audit


@pytest.mark.parametrize('checker', [check_reference_completion_study, check_reference_delta_study, check_reference_admission_study, check_multiview_completion_study, check_target_redecode_study, check_score_precision_audit])
def test_inspection_handles_planned_running_and_failed_before_summary(tmp_path, checker):
    study = tmp_path / 'study'; item = dict(run='attempt', id='test')
    assert checker(tmp_path, study, item, [])['status'] == 'planned'
    (study / 'attempt').mkdir(parents=True)
    assert checker(tmp_path, study, item, [])['status'] == 'in_progress'
    (study / 'attempt/failure.json').write_text('{}')
    result = checker(tmp_path, study, item, [])
    assert result['status'] == 'failed' and not result['complete']
