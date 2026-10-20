import numpy as np
from radio_gs.querying.response_supervision import heldout_views, incumbent_first, calibration_groups, coverage_reference_views
from radio_gs.querying.response_replacement import calibrate_margin


def test_reference_neighbours_removed_from_every_retained_view():
    class Sensor:
        def exclude(self, refs):
            removed = sorted({f for r in refs for f in range(max(0, r-1), min(20, r+2))})
            return np.setdiff1d(np.arange(20), removed), np.array(removed)
    refs, retained, excluded = heldout_views(Sensor(), np.arange(20), [.125, .375, .625, .875], .5)
    assert len(refs) >= 1 and len(retained) >= 10
    assert not set(retained)&set(excluded)
    assert all(r in excluded for r in refs)


def test_text_incumbent_reorder_preserves_all_candidate_correspondences():
    row = dict(seeds=[7, 7, 9, 9], providers=['direct', 'scope', 'direct', 'scope'],
               ious=[.1, .2, .3, .4], features=[[i] for i in range(4)],
               permuted_features=[[10+i] for i in range(4)], anchors=[3, None, 4, None])
    result = incumbent_first(row, 'scope')
    assert result['ious'] == [.2, .1, .3, .4]
    assert result['features'] == [[1], [0], [2], [3]]
    assert result['permuted_features'] == [[11], [10], [12], [13]]
    assert result['seeds'] == [7, 7, 9, 9] and result['anchors'] == [None, 3, 4, None]
    assert row['ious'] == [.1, .2, .3, .4]


def test_one_shared_margin_cannot_hide_image_harm_behind_text_gain():
    rows = [dict(scene='a', modality='text', gains=[0, .3], ious=[0, .9], evidence_available=True),
            dict(scene='a', modality='image', gains=[0, .3], ious=[.5, .4], evidence_available=True)]
    pooled, _ = calibrate_margin(rows)
    grouped, _ = calibrate_margin(calibration_groups(rows))
    assert pooled['margin'] == 0.
    assert grouped['margin'] is None


def test_coverage_constructor_finds_targets_missed_by_uniform_reference_times():
    from scipy import sparse
    class Sensor:
        def exclude(self, refs):
            return np.setdiff1d(np.arange(12), refs), np.array(refs)
    # Target 1 is only observed at late frames; target 2 only early frames.
    labels = np.repeat([1, 2], 30)
    visible = np.zeros((60, 12)); visible[:30, 7:11] = 1; visible[30:, 1:5] = 1
    config = dict(minimum_retained_view_fraction=.5, reference_views=4, minimum_target_mapping_views=2)
    refs, keep, excluded = coverage_reference_views(Sensor(), range(12), sparse.csr_matrix(visible), labels, [1, 2], config)
    assert len(refs) == 2
    assert any(f in range(7, 11) for f in refs) and any(f in range(1, 5) for f in refs)
    assert not set(keep)&set(excluded)


def test_streaming_original_local_teacher_equals_full_block_diagonal_map():
    from scipy import sparse
    from radio_gs.querying.local_appearance_match import match_regions
    rng = np.random.default_rng(91)
    features = rng.normal(size=(9, 5)).astype(np.float16)
    weights = [sparse.csr_matrix([[1., 0., .5, 0.], [0., 1., 0., 1.]], dtype=np.float32),
               sparse.csr_matrix([[1., 0., 0., .2, 0.], [0., 1., 1., 0., .3]], dtype=np.float32)]
    q = [rng.normal(size=(3, 5)).astype(np.float32), rng.normal(size=(2, 5)).astype(np.float32)]
    mass = [np.ones(3), np.array([1., .5])]
    whole = match_regions(features, sparse.block_diag(weights, format='csr'), [0, 4, 9], [0, 2, 4], q, mass, device='cpu')
    parts = [match_regions(features[a:b], w, [0, b-a], [0, 2], q, mass, device='cpu')
             for (a, b), w in zip([(0, 4), (4, 9)], weights)]
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate([p[key] for p in parts]))
