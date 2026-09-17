import pytest
from radio_gs.geometry.quality_gate import joint_scale_quality_gate


def metrics():return dict(coverage=.9,absrel=.04,free_space_fraction=.2,depth_agreement_fraction=.7,agreement_coverage=.63)


def test_improved_correspondences_do_not_override_worse_rendered_geometry():
    candidate={**metrics(),'coverage':.91,'absrel':.044,'free_space_fraction':.41}
    gate=joint_scale_quality_gate(metrics(),candidate,correspondence_accepted=True)
    assert not gate['accepted'] and gate['failed_checks']==['absrel','free_space_fraction']


@pytest.mark.parametrize('accepted',[False,True])
def test_render_gate_also_requires_correspondence_acceptance(accepted):
    assert joint_scale_quality_gate(metrics(),metrics(),correspondence_accepted=accepted)['accepted']==accepted


def test_nonfinite_validation_fails_closed():
    assert not joint_scale_quality_gate(metrics(),{**metrics(),'absrel':float('nan')},correspondence_accepted=True)['accepted']
