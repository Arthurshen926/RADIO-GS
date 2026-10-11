import torch
from radio_gs.query_field.extent_preparation import ExtentPreparation
from radio_gs.query_field.scope_completion import complete_other_view_positive, crossed_scope


def scope(p, n, c=1.):
    p = torch.tensor(p, dtype=torch.bool)
    return dict(target=p.float(), positive=p, negative=torch.tensor(n, dtype=torch.bool),
                confidence=torch.tensor(float(c)))


def prepare(matrices):
    state = {}
    for vi, matrix in enumerate(matrices):
        a = torch.tensor(matrix, dtype=torch.float32).to_sparse().coalesce()
        state.update({f'views.{vi}.'+k: v for k,v in dict(index=a.indices(), weight=a.values(),
            valid=torch.ones(a.shape[0], dtype=torch.bool), footprint=torch.ones(a.shape[1])).items()})
    return ExtentPreparation(state, a.shape[1])


def test_no_self_witness_or_recycled_completions():
    prep = prepare([[[1.], [1.]]])
    base = [scope([1,0], [0,0])]; native = [scope([1,1], [0,0])]
    out, info = complete_other_view_positive(prep, base, native)
    assert torch.equal(out[0]['positive'], base[0]['positive'])
    assert info[0]['donor_views'] == 0 and info[0]['added_positive_rays'] == 0
    # A -> B -> C must not propagate twice within a single call.
    prep = prepare([[[1.,0.]], [[.6,.4]], [[0.,1.]]])
    base = [scope([1],[0]), scope([0],[0]), scope([0],[0])]
    native = [scope([1],[0]) for _ in base]
    out, _ = complete_other_view_positive(prep, base, native)
    assert out[1]['positive'].item() and not out[2]['positive'].item()


def test_other_view_majority_requires_full_mass_and_current_hypothesis():
    prep = prepare([[[1.,0.]], [[.1,.9], [.6,.4], [.8,.2]]])
    base = [scope([1],[0]), scope([0,0,0],[0,0,0])]
    native = [scope([1],[0]), scope([1,1,0],[0,0,0])]
    out, _ = complete_other_view_positive(prep, base, native)
    assert out[1]['positive'].tolist() == [False,True,False]


def test_origin_negative_permission_survives_new_positive_activation():
    prep = prepare([[[1.,0.]], [[1.,0.],[0.,1.]]])
    base = [scope([1],[0]), scope([0,0],[0,1])]
    native = [scope([1],[0]), scope([1,0],[0,1])]
    out, info = complete_other_view_positive(prep, base, native)
    assert info[1]['newly_active'] and not out[1]['negative'].any()
    crossed, _ = crossed_scope(native[1], base[1], native[1]['confidence'])
    assert not crossed['negative'].any()


def test_foreign_negative_veto_ties_unknown_and_fixed_known_evidence():
    prep = prepare([[[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]]])
    base = [scope([1,0],[0,0]), scope([0,1],[1,0]), scope([0,0],[0,0])]
    native = [scope([1,1],[0,0]) for _ in base]
    out, _ = complete_other_view_positive(prep, base, native)
    assert not out[2]['positive'][0]  # equal foreign positive/negative mass
    assert out[2]['positive'][1]
    assert out[0]['positive'][0] and out[1]['negative'][0]  # known labels unchanged
