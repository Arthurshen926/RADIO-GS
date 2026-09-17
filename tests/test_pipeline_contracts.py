import copy
import pytest
import torch
from radio_gs.training.pipeline_contracts import validate_grounded_supervision,validate_contributions

def valid_bundle():
 return dict(schema='grounded_text_supervision.v2',negative_policy='same_view_positive_detection_required',queries=['cup'],train_frames=[10],tasks=[dict(split='train',modality='text',target_view=0,text_index=0,grounded_text_task=0,target_region=0,query_region=0)],positive_pixels=torch.tensor([[True,False]]),negative_pixels=torch.tensor([[False,True]]),positive_elements=torch.tensor([[True,False]]),negative_elements=torch.tensor([[False,True]]))
def check(b):return validate_grounded_supervision(b,frames=[10,20],train_frames=[10],proposal_views=torch.tensor([0,1]),num_elements=2,num_pixels=2,queries=['cup'])
def test_valid_grounding_contract():assert check(valid_bundle())['train_only']
@pytest.mark.parametrize('mutation',['old_schema','policy','overlap','dtype','index','validation','region'])
def test_rejects_invalid_grounding(mutation):
 b=valid_bundle()
 if mutation=='old_schema':b['schema']='grounded_text_supervision.v1'
 if mutation=='policy':b['negative_policy']='absence_is_negative'
 if mutation=='overlap':b['negative_pixels'][0,0]=True
 if mutation=='dtype':b['positive_elements']=b['positive_elements'].float()
 if mutation=='index':b['tasks'][0]['grounded_text_task']=1
 if mutation=='validation':b['tasks'][0]['target_view']=1
 if mutation=='region':b['tasks'][0]['target_region']=1
 with pytest.raises(ValueError):check(b)
@pytest.mark.parametrize('weights,ids',[(torch.tensor([float('nan')]),torch.tensor([0])),(torch.tensor([-1.]),torch.tensor([0])),(torch.ones(1),torch.tensor([2]))])
def test_invalid_contributions(weights,ids):
 with pytest.raises(ValueError):validate_contributions(torch.tensor([0]),ids,weights,num_pixels=2,num_elements=2)
def test_valid_empty_contributions():validate_contributions(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.empty(0),num_pixels=2,num_elements=2)

@pytest.mark.parametrize('error',['frame_order','mask_shape','query_nan','seed_range'])
def test_training_axis_corruption_is_rejected(error):
 from radio_gs.training.pipeline_contracts import validate_training_axes
 authority={'num_gaussians':2,'frame_indices':[10,20],'metadata':{'feature_height':1,'feature_width':2}}
 info={'frames':[10,20]}
 data={'proposal_views':torch.tensor([0,1]),'seen':torch.ones(2,2,dtype=torch.bool),'masks':torch.ones(2,1,2,dtype=torch.bool),'region_probability':torch.ones(2,2),'queries':{'image':torch.ones(2,3),'prompt':torch.ones(2,4),'text':torch.ones(1,3),'queries':['cup'],'seed_pixels':torch.tensor([[0,1],[0,1]])}}
 assert validate_training_axes(data,authority,info)['axes_valid']
 if error=='frame_order':authority['frame_indices']=[20,10]
 if error=='mask_shape':data['masks']=data['masks'].flatten(1)
 if error=='query_nan':data['queries']['image'][0,0]=float('nan')
 if error=='seed_range':data['queries']['seed_pixels'][0,0]=2
 with pytest.raises(ValueError):validate_training_axes(data,authority,info)

@pytest.mark.parametrize('corruption',[None,'outside','zero','basis'])
def test_native_element_teacher_contract(corruption):
 from radio_gs.training.pipeline_contracts import validate_native_element_teacher
 b={'schema':'consistent_native_element_teacher.v1','benchmark_data_used':False,'directions':torch.tensor([[1.,0.],[0.,0.]]),'basis':torch.eye(2),'valid':torch.tensor([True,False])}
 if corruption=='outside':b['directions'][1,0]=1
 if corruption=='zero':b['directions'][0,0]=0
 if corruption=='basis':b['basis'][0]*=2
 args=dict(num_elements=2,dimensions=2,query_dimension=2,observed=torch.tensor([True,False]))
 if corruption is None:assert validate_native_element_teacher(b,**args)['source_observed_only']
 else:
  with pytest.raises(ValueError):validate_native_element_teacher(b,**args)

def test_explicit_inactive_prompt_is_not_a_valid_seed_or_task():
 from radio_gs.training.pipeline_contracts import validate_training_axes
 authority={'num_gaussians':2,'frame_indices':[10],'metadata':{'feature_height':1,'feature_width':2}}
 info={'frames':[10]}
 data={'proposal_views':torch.tensor([0,0]),'seen':torch.ones(1,2,dtype=torch.bool),'masks':torch.ones(2,1,2,dtype=torch.bool),'region_probability':torch.ones(2,2),'queries':{'image':torch.ones(2,3),'prompt':torch.tensor([[0.,0.],[1.,1.]]),'text':torch.ones(1,3),'queries':['cup'],'seed_pixels':torch.tensor([[-1,-1],[0,1]]),'seed_rows':torch.tensor([[-1,-1],[0,1]]),'prompt_valid':torch.tensor([False,True])},'prompt_admission':{'rejected_regions':[{'region':0,'reason':'missing_positive_support'}]},'episodes':[]}
 assert validate_training_axes(data,authority,info)['axes_valid']
 data['episodes']=[{'modality':'prompt','query_region':0}]
 with pytest.raises(ValueError,match='inactive prompt used'):validate_training_axes(data,authority,info)
 data['episodes']=[];data['queries']['seed_pixels'][0]=0
 with pytest.raises(ValueError,match='sentinel'):validate_training_axes(data,authority,info)
