import numpy as np
import pytest
from PIL import Image
from radio_gs.querying.interactive_2d import validate_2d_prompt,scribble_points,select_prompt_mask,masked_query_crop

def test_pixel_coordinates_and_signs_are_explicit():
 assert validate_2d_prompt(20,10,points=[[19,9],[0,0]],labels=[1,0])['point_coords'].tolist()==[[19,9],[0,0]]
 for p,l in [([[20,9]],[1]),([[1,1],[1,1]],[1,0]),([[2,2]],[0])]:
  with pytest.raises(ValueError):validate_2d_prompt(20,10,points=p,labels=l)
 with pytest.raises(ValueError,match='composition'):validate_2d_prompt(20,10,points=[[1,1]],labels=[1],box=[0,0,10,10])
 assert validate_2d_prompt(20,10,box=[0,0,20,10])['box'].tolist()==[0,0,20,10]

def test_scribble_sampling_preserves_coordinate_order_and_rejects_conflicts():
 p=np.zeros((4,7),bool);n=p.copy();p[1,5]=True;n[3,0]=True
 xy,signs=scribble_points(p,n);assert xy.tolist()==[[5,1],[0,3]] and signs.tolist()==[1,0]
 with pytest.raises(ValueError):scribble_points(p,p)

def test_mask_selection_uses_full_signed_evidence_not_just_sam_score():
 p=np.array([[1,0],[0,0]],bool);n=np.array([[0,0],[0,1]],bool);m,r=select_prompt_mask(np.stack([np.ones((2,2),bool),p]),[.99,.3],positive=p,negative=n)
 assert np.array_equal(m,p) and r['selected']['all_signed_pixels_satisfied']
 crop,box=masked_query_crop(Image.new('RGB',(2,2),'white'),p,padding=1);assert box==(0,0,2,2) and np.asarray(crop)[1,1].tolist()==[127]*3

def test_unresolved_signed_evidence_is_reported_not_clamped():
 p=np.array([[1,0]],bool);n=~p;m,r=select_prompt_mask(np.zeros((1,1,2),bool),[.8],positive=p,negative=n)
 assert not m.any() and r['selected']['all_signed_pixels_satisfied'] is False
