import numpy as np
import torch
from torch.nn import functional as F

from scripts.research.compile_tmm_category_native import stored_native_cosines
from radio_gs.querying.category_response_supervision import native_menu, confidence_ratio, complete_candidates


def test_native_menu_matches_original_entropy_readout_with_uniform_and_missing_support():
    from scipy import sparse
    from radio_gs.querying.observation_reliability import reliable_observation_response
    transfer = np.array([[.2, .8, 0], [0, 0, 1], [0, 0, 0]], np.float32)
    cosine = np.array([[.1, .12, .13], [.2, .13, .15], [.1, .1, .1]], np.float32)
    response = torch.softmax(100*torch.from_numpy(cosine), 1)
    operator = torch.from_numpy(transfer).to_sparse()
    expected, _ = reliable_observation_response(operator, response, operator @ response, {}, 'entropy_information')
    actual, supported = native_menu(sparse.csr_matrix(transfer), cosine, np.arange(3))
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-7)
    assert supported.tolist() == [True, True, False]
    subset, _ = native_menu(sparse.csr_matrix(transfer), cosine, np.array([1]))
    np.testing.assert_allclose(subset[0], response[1], atol=2e-7)


def test_complete_menu_preserves_original_detector_mass_fusion_and_fallback(tmp_path):
    from radio_gs.querying.local_element_fusion import compile_local_evidence
    from radio_gs.querying.typed_evidence import visible_operator, view_evidence
    prior = torch.tensor([[.1,.2,.7], [.4,.3,.3], [.2,.7,.1], [.2,.2,.6]])
    auxiliary = prior.flip(1)
    observed = torch.ones(4, dtype=torch.bool)
    hit = dict(pixel_ids=torch.arange(9), gaussian_ids=torch.arange(9)%3, base_weights=torch.ones(9))
    hit_path = tmp_path/'hits.pt'; torch.save(hit, hit_path)
    views, records = [], []
    num, mass, vis = [torch.zeros_like(prior) for _ in range(3)]
    for v in range(2):
        rows = [dict(scores=torch.tensor([.8,.6]), masks=torch.full((2,3,3), p),
                     foreground_margin=torch.tensor([.12,-.03])*(v+1)) for p in (.8,.2,.6)]
        path = tmp_path/f'identity_{v}.pt'
        torch.save(dict(queries=['a','b','c'], instances=rows), path)
        views.append(dict(frame=v, identity=str(path), contributions=str(hit_path)))
        op, evop, visible = visible_operator(hit, observed, 9, False)
        target = torch.stack([r['masks'].amax(0) for r in rows], -1).reshape(9,3)
        detector = torch.tensor([float(r['scores'].max()) for r in rows])
        e = view_evidence(op, evop, visible, target, detector>0, v, 3, 3)
        known = (e.positive|e.negative).float()
        num += e.adjoint @ (known*detector*target)
        mass += e.adjoint @ (known*detector)
        vis += e.adjoint @ e.visible.float()
        records.append(dict(identity=[dict(score=r['scores'].tolist(), margin=r['foreground_margin'].tolist(),
            known_pixels=int(known[:,c].sum())) for c,r in enumerate(rows)]))
    original = compile_local_evidence(prior.logit(), observed, {'source_evidence':dict(raster_hw=[3,3],views=views)},
                                      ['a','b','c'], auxiliary.logit())
    ratio = confidence_ratio(records, [0,1], 3)
    candidates = complete_candidates(prior.numpy(), auxiliary.numpy(), num.numpy(), mass.numpy(), vis.numpy(), ratio)
    np.testing.assert_allclose(candidates[0], original['average_logits'].sigmoid(), atol=2e-7)
    np.testing.assert_array_equal(candidates[0,3], prior[3])
    np.testing.assert_allclose(candidates[1], (prior+auxiliary)*.5, atol=2e-7)
    np.testing.assert_allclose(candidates[2], original['features'][:,:,4], atol=2e-7)


def test_source_transport_rejects_truncation_and_oversized_payload():
    import io
    import struct
    import pytest
    from scripts.research.compile_tmm_category_utility import receive
    payload = io.BytesIO(); torch.save({'scene':'s', 'frame':3, 'rows':[]}, payload)
    data = payload.getvalue()
    assert receive(io.BytesIO(struct.pack('<Q',len(data))+data))['frame'] == 3
    with pytest.raises(RuntimeError):
        receive(io.BytesIO(struct.pack('<Q',len(data))+data[:10]))
    with pytest.raises(ValueError):
        receive(io.BytesIO(struct.pack('<Q',33*2**20)))


def test_class_response_cache_preserves_production_quantization_and_menu_order():
    generator = torch.Generator().manual_seed(43)
    descriptor = F.normalize(torch.randn(11, 768, generator=generator), dim=-1)
    query = F.normalize(torch.randn(19, 768, generator=generator), dim=-1)
    # The original field writer stores descriptors in fp16; the runtime also
    # round-trips query encodings through fp16, then renormalizes both.
    original = F.normalize(descriptor.half().float(), dim=-1) @ F.normalize(query.half().float(), dim=-1).T
    cached = stored_native_cosines(descriptor, query)
    torch.testing.assert_close(cached, original, rtol=0, atol=0)
    order = torch.randperm(19, generator=generator)
    torch.testing.assert_close(stored_native_cosines(descriptor, query[order]), cached[:, order])
    assert np.array_equal((100*cached).softmax(1).argmax(1), (100*original).softmax(1).argmax(1))
