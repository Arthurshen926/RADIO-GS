import pytest
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField

def test_teacher_free_all_modalities_and_query_order():
    torch.manual_seed(4)
    model=ElementRetrievalField(10,8,{"text":5,"image":7,"prompt":3})
    for mode,dim in (("text",5),("image",7),("prompt",3)):
        q=torch.randn(4,dim)
        result=model.retrieve(q,mode)
        assert not result["valid"].any()
        assert torch.allclose(model.logits(q[[2,0]],mode),model.logits(q,mode)[:,[2,0]],atol=1e-6)
        model.logits(q,mode).square().mean().backward()
        assert model.query_adapters[mode].weight.grad.abs().sum()>0
    assert model.latent.grad.abs().sum()>0
    clone=ElementRetrievalField(10,8,{"text":5,"image":7,"prompt":3})
    clone.load_state_dict(model.state_dict())
    assert torch.equal(clone.latent,model.latent)

def test_query_adapter_contract():
    m=ElementRetrievalField(2,4,{"image":3})
    with pytest.raises(ValueError):m.logits(torch.ones(1,3),"text")
    with pytest.raises(ValueError):m.logits(torch.full((1,3),float("nan")),"image")


def test_cold_field_binds_geometry_and_query_encoder(tmp_path):
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    from radio_gs.models.element_retrieval_field import load_element_retrieval_field
    from radio_gs.utils.immutable_artifacts import sha256_file
    geometry=FrozenGaussianGeometry(torch.tensor([[0.,0.,1.],[1.,0.,1.]]),
        torch.tensor([[0.,0.,1.],[0.,0.,1.]]),tangent_sigma=.01,normal_sigma=.001)
    architecture={"num_elements":2,"latent_dim":4,"query_dims":{"image":3}}
    model=ElementRetrievalField(**architecture)
    identity={"model":"teacher-a","sha256":"a"*64,"normalization":"l2_native_patch"}
    path=tmp_path/'field.pt'
    torch.save({"schema":"radio_gs.element_retrieval_field.candidate.v1",
        "architecture":architecture,"state_dict":model.state_dict(),
        "geometry_binding":geometry.fingerprint(),"query_encoder":identity},path)
    kwargs=dict(expected_sha256=sha256_file(path),geometry=geometry,expected_query_encoder=identity)
    restored,_=load_element_retrieval_field(path,**kwargs)
    assert torch.equal(restored.latent,model.latent)
    with pytest.raises(ValueError,match="encoder identity"):
        load_element_retrieval_field(path,**{**kwargs,"expected_query_encoder":{**identity,"model":"teacher-b"}})
    geometry.scales.mul_(2)
    with pytest.raises(ValueError,match="geometry changed"):
        load_element_retrieval_field(path,**kwargs)


def test_explicit_aligned_query_encoders_share_projection_and_gradient():
    field=ElementRetrievalField(4,3,{'text':5,'image':5,'prompt':7},shared_query_groups=[['text','image']])
    field.observed.fill_(True)
    q=torch.randn(1,5)
    assert torch.equal(field.retrieve(q,'text')['scores'],field.retrieve(q,'image')['scores'])
    field.logits(q,'image').sum().backward()
    assert field.query_adapters['image'].weight is field.query_adapters['text'].weight
    assert field.query_adapters['text'].weight.grad is not None


def test_shared_projection_rejects_incompatible_dimensions():
    import pytest
    with pytest.raises(ValueError,match='equal dimensions'):
        ElementRetrievalField(4,3,{'text':5,'image':6},shared_query_groups=[['text','image']])


def test_unobserved_features_cannot_win_retrieval_in_either_query_path():
    from radio_gs.models.signed_query_refinement import retrieve_latent_query
    field=ElementRetrievalField(2,2,{'image':2})
    with torch.no_grad():
        field.query_adapters['image'].weight.copy_(torch.eye(2))
        field.latent.copy_(torch.tensor([[1.,0.],[0.,1.]]))
        field.observed.copy_(torch.tensor([False,True]))
    query=torch.tensor([[1.,0.]])
    ordinary=field.retrieve(query,'image')
    signed=retrieve_latent_query(field,query)
    assert ordinary['scores'][0].item()==signed['scores'][0].item()==0
    assert ordinary['scores'][1].item()>0
    assert not ordinary['valid'][0]
    assert torch.equal(field.retrieve(query,'image',rows=torch.tensor([1,0]))['scores'],ordinary['scores'][[1,0]])


def test_contrastive_text_uses_same_element_space_and_respects_unknown():
    from radio_gs.models.element_retrieval_field import retrieve_contrastive_text
    field=ElementRetrievalField(3,2,{'text':2})
    with torch.no_grad():
        field.query_adapters['text'].weight.copy_(torch.eye(2))
        field.latent.copy_(torch.tensor([[1.,0.],[0.,1.],[1.,0.]]))
        field.observed.copy_(torch.tensor([True,True,False]))
    result=retrieve_contrastive_text(field,torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]]))
    assert result['scores'][0,0]>.99 and result['scores'][1,0]<.01
    assert result['scores'][2,0]==0 and not result['valid'][2]
    assert torch.equal(retrieve_contrastive_text(field,torch.tensor([[1.,0.]]),torch.tensor([[1.,0.]]))['scores'][:2],torch.full((2,1),.5))


def test_calibration_preserves_cosine_order_and_legacy_initial_values():
 model=ElementRetrievalField(3,2,{'text':2},calibrated_readout=True)
 cosine=torch.tensor([-.5,0.,.5])
 assert torch.allclose(model.cosine_logits(cosine,'text'),10*cosine-5)
 with torch.no_grad():model.readout_bias['text'].fill_(-1.)
 assert torch.equal(model.cosine_logits(cosine,'text').argsort(),cosine.argsort())
 assert model.cosine_logits(torch.tensor(.1),'text').abs()<1e-6


def test_calibrated_prompt_refinement_uses_same_readout_without_field_updates():
 from radio_gs.models.signed_query_refinement import refine_signed_query,retrieve_latent_query
 model=ElementRetrievalField(2,2,{'prompt':2},calibrated_readout=True)
 with torch.no_grad():
  model.latent.copy_(torch.eye(2));model.query_adapters['prompt'].weight.copy_(torch.eye(2));model.observed.fill_(True);model.readout_bias['prompt'].fill_(-1.)
 q=torch.tensor([[1.,0.]])
 result=refine_signed_query(model,q,[torch.tensor([0]),torch.tensor([1])],[torch.ones(1),torch.ones(1)],[1.,0.],only_if_violated=False,steps=2)
 assert torch.allclose(result['seed_before'],model.retrieve(q,'prompt')['scores'][:,0])
 assert torch.allclose(result['seed_after'],retrieve_latent_query(model,result['latent_query'])['scores'][:,0])
 assert all(p.grad is None for p in model.parameters())
