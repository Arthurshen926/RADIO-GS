import pytest
from radio_gs.geometry.frozen_depth_contract import effective_mapping_depth_affine


def test_final_scale_applies_to_affine_offset_too():
    a, b = effective_mapping_depth_affine(7, dict(accepted=True, scale=2., offset=.3),
                                          None, [7], [1.5], 10.)
    assert (a, b) == pytest.approx((30., .45))


def test_recovery_does_not_reuse_rejected_affine():
    a, b = effective_mapping_depth_affine(22, dict(accepted=False, scale=99., offset=77.),
                                          dict(frame=22, accepted=True, scale=.5), [7, 22], [9., 2.], 10.)
    assert (a, b) == (10., 0.)


@pytest.mark.parametrize('frames,scales,recovery', [
    ([7], [1.], dict(frame=22, accepted=True, scale=1.)),
    ([22, 22], [1., 1.], dict(frame=22, accepted=True, scale=1.)),
    ([22], [1.], dict(frame=7, accepted=True, scale=1.)),
    ([22], [1.], None),
    ([22], [float('nan')], dict(frame=22, accepted=True, scale=1.)),
])
def test_invalid_or_missing_evidence_fails_closed(frames, scales, recovery):
    with pytest.raises(ValueError):
        effective_mapping_depth_affine(22, dict(accepted=False), recovery, frames, scales, 10.)
