from PIL import Image
import pytest
import torch
from radio_gs.utils.query_input_identity import query_call_sha256


def test_image_pixels_and_encoder_are_bound():
    im=Image.new('RGB',(3,3));a=query_call_sha256('image',[im],{}, {'version':1});im.putpixel((0,0),(1,0,0))
    assert a!=query_call_sha256('image',[im],{}, {'version':1})
    assert a!=query_call_sha256('image',[Image.new('RGB',(3,3))],{}, {'version':2})


def test_registered_prompt_file_and_coordinates_are_bound(tmp_path):
    p=tmp_path/'image.png';Image.new('RGB',(2,2)).save(p);args=[str(p),[0,1],torch.zeros(2,3)]
    h=query_call_sha256('prompt',args,{},{});args[2][0,0]=1
    assert h!=query_call_sha256('prompt',args,{},{})
    args[2].zero_();Image.new('RGB',(2,2),color='red').save(p)
    assert h!=query_call_sha256('prompt',args,{},{})
    with pytest.raises(TypeError):query_call_sha256('text',[object()],{},{})
