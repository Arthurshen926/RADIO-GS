import pytest
import torch
from radio_gs.querying.registered_coordinate_query import encode_registered_coordinates, coordinate_only_prompt_architecture


def test_preserves_signed_coordinates_without_visual_encoder():
    xyz=torch.tensor([[1.,2.,3.],[-1.,0.,1.]])
    lo=torch.tensor([-2.,-2.,-2.]);extent=torch.tensor([4.,8.,6.])
    q=encode_registered_coordinates(xyz,lo,extent,1542)
    assert not q[0,:-6].any()
    torch.testing.assert_close(q[0,-6:].reshape(2,3)*extent+lo,xyz)
    assert not coordinate_only_prompt_architecture({'scale_affinity':None})
    assert coordinate_only_prompt_architecture({'scale_affinity':{'scales':[1.]}})


def test_invalid_registration_fails():
    with pytest.raises(ValueError):encode_registered_coordinates(torch.zeros(2,3),torch.zeros(3),torch.zeros(3),6)
