import numpy as np
import pytest
from radio_gs.querying.reference_completion import signed_scribble_points, enforce_reference_scribbles


def test_signed_points_are_on_original_scribbles_and_cover_disconnected_support():
    p = np.zeros((14, 20), bool); n = p.copy()
    p[1:3, 2:5] = True; p[11:13, 16:19] = True; n[6:8, 9:12] = True
    r = signed_scribble_points(p, n, 6)
    xy, labels = r['point_coords'].astype(int), r['point_labels']
    assert len(xy) == len(set(map(tuple, xy))) == 12
    assert p[xy[labels == 1, 1], xy[labels == 1, 0]].all()
    assert n[xy[labels == 0, 1], xy[labels == 0, 0]].all()
    assert (xy[labels == 1, 0] < 5).any() and (xy[labels == 1, 0] > 15).any()
    again = signed_scribble_points(p, n, 6)
    assert all(np.array_equal(r[k], again[k]) for k in r)


def test_only_explicit_reference_constraints_override_response():
    x = np.array([[.2, .8, .7], [.9, .1, .3]], np.float32)
    p = np.zeros_like(x, dtype=bool); n = p.copy(); p[0, 0] = True; n[1, 0] = True
    out = enforce_reference_scribbles(x, p, n)
    assert out[0, 0] == 1 and out[1, 0] == 0
    assert np.array_equal(out[~(p | n)], x[~(p | n)]) and x[0, 0] == np.float32(.2)
    with pytest.raises(ValueError): enforce_reference_scribbles(x, p, p)
    with pytest.raises(ValueError): signed_scribble_points(p, np.zeros_like(n))
