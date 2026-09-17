import struct
import numpy as np
import pytest
from radio_gs.geometry.source_camera_assets import source_colmap_assets


def setup(root, *, width=100, model=1):
    np.save(root/'hwf_cxcy.npy',np.array([80,100,90,90,50,40]))
    sparse=root/'dense/sparse';sparse.mkdir(parents=True)
    (sparse/'images.bin').touch();(sparse/'points3D.bin').touch()
    params=[90,90,50,40] if model==1 else [90,50,40,.1]
    (sparse/'cameras.bin').write_bytes(struct.pack('<QiiQQ4d',1,1,model,width,80,*params))
    return sparse


def test_released_rgb_uses_full_matching_dense_model(tmp_path):
    sparse=setup(tmp_path)
    assert source_colmap_assets(tmp_path)==(sparse,tmp_path/'hwf_cxcy.npy')


@pytest.mark.parametrize('options',[{'width':99},{'model':2}])
def test_incompatible_or_distorted_camera_is_rejected(tmp_path,options):
    setup(tmp_path,**options)
    with pytest.raises(ValueError):source_colmap_assets(tmp_path)


def test_sidecar_never_silently_falls_back_to_original_camera(tmp_path):
    np.save(tmp_path/'hwf_cxcy.npy',[80,100,90,90,50,40])
    with pytest.raises(ValueError,match='matching dense'):source_colmap_assets(tmp_path)


def test_legacy_source_layout_is_unchanged_without_sidecar(tmp_path):
    assert source_colmap_assets(tmp_path)==(tmp_path/'sparse/0',None)
