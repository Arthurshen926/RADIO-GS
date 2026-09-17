import pytest
import torch
from radio_gs.training.reliable_correspondence import reliable_correspondence_tasks

def task(q,t,m='image',neg=None):
 return {'query_region':q,'target_region':t,'target_view':t,'negative_regions':neg or [],'split':'train','modality':m}

def indexed(pairs):
 return list(enumerate([task(q,t,m) for q,t in pairs for m in ('image','prompt')]))

def test_rejects_track_closing_onto_a_second_region_in_same_view():
 tasks=indexed([(0,1),(1,2),(2,3)]);p=torch.ones(4,8);v=torch.tensor([0,1,2,0]);seen=torch.ones(3,8,dtype=torch.bool)
 result,audit=reliable_correspondence_tasks(tasks,p,v,seen)
 assert audit['accepted_pairs']==2
 assert audit['rejected_edges']==[{'pair':[2,3],'reason':'same_view_track_collision'}]
 assert len(result)==4 and tasks[-1][1]['negative_regions']==[]

def test_containment_is_not_full_extent_correspondence_and_orphan_negative_removed():
 p=torch.tensor([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,0]],dtype=torch.float)
 tasks=indexed([(0,1),(2,3)]);tasks[0][1]['negative_regions']=[3]
 result,a=reliable_correspondence_tasks(tasks,p,torch.tensor([0,1,0,1]),torch.ones(2,4,dtype=torch.bool),minimum_coverage=.75)
 assert a['accepted_pairs']==1 and a['removed_negative_region_entries']==1
 assert result[0][1]['negative_regions']==[] and tasks[0][1]['negative_regions']==[3]

def test_validation_cannot_admit_training_pairs():
 tasks=indexed([(0,1)]);tasks[0][1]['split']='validation'
 with pytest.raises(ValueError,match='training-only'):
  reliable_correspondence_tasks(tasks,torch.ones(2,2),torch.tensor([0,1]),torch.ones(2,2,dtype=torch.bool))

def test_source_tasks_cover_unmatched_regions_without_validation_or_outside_negatives():
 from radio_gs.training.reliable_correspondence import source_region_query_tasks
 data={'proposal_views':torch.tensor([0,0,0,1]),'masks':torch.tensor([[[1,0,0]],[[0,1,0]],[[1,0,0]],[[0,0,1]]],dtype=torch.bool),'queries':{'image':torch.eye(4),'region_dino':torch.eye(4)},'episodes':[]}
 out,audit=source_region_query_tasks([],data,[0]);assert len(out)==6 and audit['regions']==3
 assert out[0][1]['negative_regions']==[1]
 assert {e['query_region'] for _,e in out}=={0,1,2}
 assert all(e['query_region']==e['target_region'] and e['target_view']==0 for _,e in out)
 assert len({i for i,_ in out})==6

def test_missing_signed_seed_keeps_image_region_and_rejects_prompt_only():
 from radio_gs.training.reliable_correspondence import source_region_query_tasks
 data={'proposal_views':torch.tensor([0,0]),'masks':torch.tensor([[[1,0]],[[0,1]]],dtype=torch.bool),'queries':{'image':torch.eye(2),'region_dino':torch.eye(2),'prompt_valid':torch.tensor([False,True])},'episodes':[]}
 out,audit=source_region_query_tasks([],data,[0])
 assert [(e['query_region'],e['modality']) for _,e in out]==[(0,'image'),(1,'image'),(1,'prompt')]
 assert audit['added_tasks_per_modality']=={'image':2,'prompt':1}
 with pytest.raises(ValueError,match='inadmissible prompt'):
  source_region_query_tasks([(0,task(0,1,'prompt'))],data,[0])

def test_no_positive_geometry_cannot_become_all_negative_image_supervision():
 from radio_gs.training.reliable_correspondence import source_region_query_tasks
 data={'proposal_views':torch.tensor([0,0]),'masks':torch.tensor([[[1,0]],[[0,1]]],dtype=torch.bool),'queries':{'image':torch.eye(2),'region_dino':torch.eye(2),'prompt_valid':torch.tensor([False,True])},'episodes':[],'prompt_admission':{'rejected_regions':[{'region':0,'reason':'missing_positive_support'}]}}
 out,audit=source_region_query_tasks([],data,[0])
 assert all(e['query_region']==1 for _,e in out)
 assert len(out)==2 and audit['rejected_no_positive_geometry_regions']==[0]
