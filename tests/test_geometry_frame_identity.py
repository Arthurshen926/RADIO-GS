import json,struct
from pathlib import Path
import pytest
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images
from radio_gs.v4.geometry.batch_moge3 import _load_source_frames
from radio_gs.utils.immutable_artifacts import sha256_file


def camera_file(path):
    names=['DJI_20200223_163022_557.JPG','DJI_20200223_163104_557.JPG']
    data=struct.pack('<Q',2)
    for i,name in enumerate(names):
        data+=struct.pack('<i4d3di',i+1,1,0,0,0,i,0,0,1)+name.encode()+b'\0'+struct.pack('<Q',0)
    path.write_bytes(data)
    return names


def test_ambiguous_numeric_suffix_cannot_silently_overwrite_camera(tmp_path):
    names=camera_file(tmp_path/'images.bin')
    cameras={1:{'model_id':1,'params':[90,90,50,40],'width':100,'height':80}}
    with pytest.raises(ValueError,match='ambiguous'):_read_images(tmp_path/'images.bin',cameras)
    views=_read_images(tmp_path/'images.bin',cameras,{names[0]:0,names[1]:1})
    assert list(views)==[0,1] and views[0].image_id!=views[1].image_id
    assert views[0].world_to_camera[0,3]!=views[1].world_to_camera[0,3]


def test_moge_explicit_identity_uses_only_sealed_sources_and_checks_hash(tmp_path):
    images=tmp_path/'images';images.mkdir()
    names=['DJI_163022_557.JPG','DJI_163104_557.JPG']
    for name in names:(images/name).write_bytes(b'source')
    target=images/'target.JPG';target.write_bytes(b'forbidden')
    authority={'metadata':{k:False for k in ['benchmark_images_opened','benchmark_masks_opened','text_queries_opened']},'frame_indices':[0,1],'camera_name_to_frame_index':{names[0]:0,names[1]:1,'target.JPG':2},'records':[{'frame_index':i,'source_image':str(images/n),'source_image_sha256':sha256_file(images/n)} for i,n in enumerate(names)]}
    path=tmp_path/'authority.json';path.write_text(json.dumps(authority))
    assert _load_source_frames(path,tmp_path)==[(i,images/n) for i,n in enumerate(names)]
    (images/names[0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):_load_source_frames(path,tmp_path)
