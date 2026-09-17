import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_decoder_readout import affine_decoder_cosine


def test_native_measured_extension_is_image_only_and_bypasses_untrained_residual():
    f=ElementRetrievalField(3,4,{'text':5,'image':5,'prompt':2},shared_query_groups=[['text','image']],
        native_siglip_readout=True,native_semantic_dim=2,native_semantic_residual={'relative_bound':.25,'modalities':['image']},image_support_extension=True)
    with torch.no_grad():
        f.observed[0]=True;f.image_observed[:2]=True;f.native_residual_decoder.weight.normal_()
    q=torch.randn(2,5)
    text=f.retrieve(q,'text');image=f.retrieve(q,'image')
    assert text['valid'].tolist()==[True,False,False]
    assert image['valid'].tolist()==[True,True,False]
    assert torch.equal(text['scores'][1:],torch.zeros(2,2))
    baseline=affine_decoder_cosine(f.latent[1:2,:2],q,f.native_decoder.weight,f.native_decoder.bias)
    assert torch.allclose(image['scores'][1:2],torch.sigmoid(10*baseline-5))
    assert torch.allclose(image['scores'][1:2],f.retrieve(q,'image',torch.tensor([1]))['scores'])
