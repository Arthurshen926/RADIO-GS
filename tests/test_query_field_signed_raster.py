import json

import numpy as np
from PIL import Image
import pytest
import torch

from radio_gs.query_field.adapters import pixel_support, raster_support, scribble_packet, raster_extent_scale
from radio_gs.query_field.public_nvos import load_strokes, score
from radio_gs.query_field.storage import sha256
from radio_gs.query_field.observations import source_sam_metadata
from test_query_preserving_field import example


def test_source_provenance_normalization_never_accepts_query_conditioned_masks():
    flat = dict(schema_version=2, source='official_sam3_interactive_grid_multimask_hierarchy',
                source_image_sha256='rgb', checkpoint_sha256='weights', query_free=True)
    nested = dict(generation_contract=dict(query_free=True), source_image=dict(sha256='rgb'),
                  checkpoint=dict(sha256='weights'))
    assert source_sam_metadata(flat) == source_sam_metadata(nested)
    for metadata in [dict(flat, query_free=False), dict(flat, source='prompted_sam'),
                     dict(nested, generation_contract=dict(query_free=False))]:
        with pytest.raises(ValueError):
            source_sam_metadata(metadata)


def operator():
    # Unequal total opacity must not give one registered pixel more evidence.
    return torch.sparse_coo_tensor(torch.tensor([[0, 0, 1], [0, 1, 2]]),
                                   torch.tensor([.2, .6, .1]), (3, 12)).coalesce()


def test_registered_rasters_preserve_ray_uncertainty_and_report_missing_mass():
    support, coverage = raster_support(operator(), torch.tensor([1., 1., 2.]))
    assert support.indices.tolist() == [0, 1, 2]
    torch.testing.assert_close(support.weights, torch.tensor([.25, .75, 1.]))
    assert coverage == .5
    # Equivalent single-pixel evidence gives exactly the same normalized support.
    single = pixel_support(operator(), 0)
    pooled, _ = raster_support(operator(), torch.tensor([3., 0., 0.]))
    torch.testing.assert_close(pooled.weights/pooled.weights.sum(), single.weights/single.weights.sum())
    for bad in [torch.zeros(3), torch.tensor([0., 0., 1.]), torch.tensor([-1., 1., 0.]),
                torch.tensor([float('nan'), 1., 0.]), torch.ones(4)]:
        with pytest.raises(ValueError):
            raster_support(operator(), bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA registration repeatability')
def test_large_raster_pooling_replays_exactly_on_cuda():
    generator = torch.Generator().manual_seed(73)
    op = torch.sparse_coo_tensor(torch.stack([torch.randint(100, (50000,), generator=generator),
                                             torch.randint(200, (50000,), generator=generator)]),
                                 torch.rand(50000, generator=generator), (100, 200)).cuda().coalesce()
    evidence = torch.rand(100, generator=generator).cuda()
    expected, coverage = raster_support(op, evidence)
    for _ in range(3):
        repeated, actual_coverage = raster_support(op, evidence)
        assert torch.equal(repeated.indices, expected.indices)
        assert torch.equal(repeated.weights, expected.weights)
        assert actual_coverage == coverage


def test_shared_signed_core_and_overlapping_uncertain_contributors():
    field = example()
    packet, receipt = scribble_packet(operator(), torch.tensor([1., 0., 0.]), torch.tensor([.5, 1., 0.]))
    packet.validate(len(field.xyz), field.semantic.native_dim, field.local.native_dim)
    expected = field.member_logits(packet.positive, field.default_scale).sigmoid()
    expected *= 1-field.member_logits(packet.negative, field.default_scale).sigmoid()
    expected[~field.observed] = 0
    torch.testing.assert_close(field.query(packet).membership_scores, expected)
    assert not packet.exact_positive and not receipt['exact_element_constraints']


def test_extent_uses_distinct_rays_and_is_invariant_to_world_units():
    op = operator()
    xyz = torch.zeros(12, 3)
    xyz[0, 0], xyz[1, 0], xyz[2, 0] = -3, 1, 2
    # First uncertain ray has centre zero; second has centre two.
    evidence = torch.tensor([1., 1., 0.])
    assert raster_extent_scale(op, evidence, xyz, 1., .5) == pytest.approx(2.)
    assert raster_extent_scale(op, evidence, xyz*4+7, 4., .5) == pytest.approx(2.)
    # Within-ray spread is not an estimate of target extent.
    assert raster_extent_scale(op, torch.tensor([1., 0., 0.]), xyz, 1., .5) == .5


def test_area_pooling_retains_thin_disjoint_strokes_even_in_same_coarse_pixel(tmp_path):
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, 0, 0] = 255
    masks[1, 1, 1] = 255
    paths = [tmp_path/'positive.png', tmp_path/'negative.png']
    for path, mask in zip(paths, masks):
        Image.fromarray(mask).save(path)
    resized, receipt = load_strokes(*paths, (2, 2))
    assert torch.equal(resized[:, 0, 0], torch.tensor([1/16, 1/16]))
    assert receipt['positive_pixels'] == 1
    Image.fromarray(masks[0]).save(paths[1])
    with pytest.raises(ValueError, match='disjoint'):
        load_strokes(*paths, (2, 2))


def test_scorer_requires_every_frozen_prediction_and_preserves_float_threshold(tmp_path):
    config = dict(jobs=[dict(name='positive'), dict(name='signed')], output_hw=[2, 2], alpha_threshold=.5)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    target = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    Image.fromarray(target).save(tmp_path/'target.png')
    scoring = dict(prediction_config=str(path), gt_mask=str(tmp_path/'target.png'))
    for job in config['jobs']:
        root = tmp_path/job['name']; root.mkdir()
        response = torch.tensor([[.5001, .5], [.4999, .8]])
        torch.save(dict(response=response, retained_response=response, selected=torch.tensor([True])),
                   root/'predictions.pt')
        receipt = dict(configuration=config, job=job, predictions_sha256=sha256(root/'predictions.pt'))
        (root/'prediction.json').write_text(json.dumps(receipt))
        if job['name'] == 'positive':
            with pytest.raises(FileNotFoundError):
                score(scoring, tmp_path)
    report = score(scoring, tmp_path)
    assert all(row['foreground_iou'] == 1 for row in report['results'])
    (tmp_path/'evaluation.json').unlink()
    # Original NVOS masks are larger than its declared prediction raster.
    Image.fromarray(np.repeat(np.repeat(target, 2, axis=0), 2, axis=1)).save(tmp_path/'target.png')
    resized = score(scoring, tmp_path)
    assert resized['target_hw'] == [4, 4] and resized['prediction_hw'] == [2, 2]
    (tmp_path/'evaluation.json').unlink()
    torch.save(dict(response=torch.ones(2, 2)), tmp_path/'signed/predictions.pt')
    with pytest.raises(ValueError, match='payload changed'):
        score(scoring, tmp_path)
