import torch
from radio_gs.geometry.fixed_feature_correspondence import fixed_feature_correspondences


def test_appearance_only_vetoes_geometrically_valid_pairs():
    h = w = 16
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    rays = torch.stack([x+.5, y+.5, torch.ones_like(x)], -1).float()
    view = dict(depth=torch.ones(h,w), valid=torch.ones(h,w,dtype=torch.bool),
                normals=torch.tensor([0.,0.,1.]).expand(h,w,3), rays=rays,
                pose=torch.eye(4), k=torch.eye(3))
    a = torch.zeros(2,h,w); a[0] = 1.
    b = a.clone(); b[0,:,8:] = 0.; b[1,:,8:] = 1.
    banks, audit = fixed_feature_correspondences([dict(view,frame=1),dict(view,frame=2)], [a,b], 1., stride=1)
    assert len(banks['geometry'][0]) == 256
    assert len(banks['feature'][0]) == 128
    assert audit[0]['feature'] <= audit[0]['geometric']
    # No invented match: accepted appearance rows are a subset of the same
    # exact camera/depth correspondences and retain their original coordinates.
    assert (banks['feature'][2][:,0] < 8).all()
    torch.testing.assert_close(banks['feature'][2], banks['feature'][3])
