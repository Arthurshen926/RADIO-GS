import copy
import torch
from radio_gs.training.partial_scope_metrics import extent_measures, aggregate, episode_rows, select_ridge, odds_probability


def example():
    return dict(visible=torch.tensor([1,1,1,0]), known=torch.tensor([[1],[1],[0],[1]]), target=torch.tensor([[1],[0],[0],[1]]))


def test_unknown_invisible_and_negative():
    e=example(); a=torch.tensor([[.8],[.9],[0.],[1.]])
    b=a.clone();b[2]=1;b[3]=0
    x,y=extent_measures(a,e),extent_measures(b,e)
    for key in ['known_iou','known_brier','positive_recall','negative_selection']:
        assert torch.equal(x[key],y[key])
    assert x['known_iou'].item()==.5 and x['negative_selection'].item()==1
    assert x['positive_count'].item()==1
    choices=[dict(ridge=r,known_brier=x['known_brier'].item(),unknown_selection=u) for r,u in [(0.1,0),(1,1),(10,1)]]
    assert select_ridge(choices,'partial_known_brier')['ridge']==10


def test_empty_and_single_class():
    e=example();e['known'].zero_();m=extent_measures(torch.zeros(4,1),e)
    assert torch.isnan(m['known_brier']).all() and torch.isnan(m['known_iou']).all()
    assert aggregate(episode_rows(m,0,[0]))['known_brier'] is None
    e['known'][0]=1;m=extent_measures(torch.zeros(4,1),e)
    assert m['class_count'].item()==1 and m['known_brier'].item()==1
    assert torch.isnan(m['negative_selection']).all()
    e['known'].zero_();e['known'][1]=1;m=extent_measures(torch.zeros(4,1),e)
    assert m['class_count'].item()==1 and m['known_brier'].item()==0
    assert torch.isnan(m['positive_recall']).all()


def test_equal_class_and_episode_weight_and_tie():
    rows=[dict(known_brier=.2,class_count=1),dict(known_brier=.8,class_count=2)]
    assert aggregate(rows)['known_brier']==.5
    choices=[dict(ridge=.1,known_brier=.2),dict(ridge=10.,known_brier=.2000009)]
    assert select_ridge(choices,'partial_known_brier')['ridge']==10
    choices[1]['known_brier']=.200002
    assert select_ridge(choices,'partial_known_brier')['ridge']==.1


def test_zero_delta_odds():
    logits=torch.tensor([-100.,-20.,0.,20.,100.])
    assert torch.equal(odds_probability(logits,torch.zeros_like(logits)),logits.sigmoid())


def test_legacy_formula_and_old_payload(monkeypatch):
    import radio_gs.querying.source_kernel_extent as module
    e=example();p=torch.tensor([[.8],[.9],[1.],[0.]])
    pred=(p>.5)&e['visible'][:,None].bool();target=e['target'].bool()
    old=(pred&target).sum(0)/(pred|target).sum(0).clamp_min(1)
    assert torch.equal(extent_measures(p,e)['legacy_visible_iou'],old)
    monkeypatch.setattr(module,'attach_kernel_native',lambda m,p:m)
    class Model:
        latent=torch.zeros(2,1)
        def logits(self,q,modality,rows=None):return torch.tensor([[-20.],[2.]])
    payload=dict(kernel_baseline='pipeline',source_kernel_extent=dict(anchors=torch.ones(1,1),inverse=torch.ones(1,1),residual=torch.zeros(2,1),concentration=20))
    m=Model();original=m.logits(None,'text');module.attach_source_kernel_extent(m,payload)
    assert torch.equal(m.logits(torch.ones(1,1),'text'),torch.logit(original.sigmoid().clamp(1e-6,1-1e-6)))
    payload['source_kernel_extent']['readout']='odds';m=Model();module.attach_source_kernel_extent(m,payload)
    assert torch.equal(m.logits(torch.ones(1,1),'text'),original)


def test_deployed_odds_matches_selection_with_serialized_state(monkeypatch):
    import radio_gs.querying.source_kernel_extent as module
    from torch.nn import functional as F
    monkeypatch.setattr(module,'attach_kernel_native',lambda m,p:m)
    class Model:
        latent=torch.zeros(3,1)
        def logits(self,q,modality,rows=None):return torch.tensor([[-20.,-1.],[0.,2.],[12.,20.]])
    torch.manual_seed(7)
    anchors=F.normalize(torch.randn(2,4),dim=1).half()
    q=F.normalize(torch.randn(2,4),dim=1)
    normalized=F.normalize(anchors.float(),dim=1)
    gram=torch.exp(20*(normalized@normalized.T-1))
    inverse=torch.linalg.inv(gram+torch.eye(2))
    residual=(torch.randn(3,2)*.03).half()
    state=dict(anchors=anchors,inverse=inverse,residual=residual,concentration=20,readout='odds')
    kernel=torch.exp(20*(normalized@F.normalize(q,dim=-1).T-1))
    expected=odds_probability(Model().logits(q,'text'),residual.float()@(inverse@kernel))
    m=Model();module.attach_source_kernel_extent(m,dict(kernel_baseline='pipeline',source_kernel_extent=state))
    assert torch.equal(m.logits(q,'text').sigmoid(),expected)


def test_no_supervision_cannot_select():
    import pytest
    with pytest.raises(ValueError,match='No valid'):
        select_ridge([dict(ridge=10,known_brier=None)],'partial_known_brier')


def test_brier_balances_classes_not_pixels():
    e=dict(visible=torch.ones(101,dtype=torch.bool),known=torch.ones(101,1),target=torch.zeros(101,1))
    e['target'][0]=1
    metrics=extent_measures(torch.zeros(101,1),e)
    assert metrics['known_brier'].item()==.5
    assert metrics['positive_brier'].item()==1 and metrics['negative_brier'].item()==0


def test_saturated_training_forward_roundtrips_final_residual(monkeypatch, tmp_path):
    from scripts.research.fit_source_kernel_extent import extent_training_states
    import radio_gs.querying.source_kernel_extent as module
    monkeypatch.setattr(module, 'attach_kernel_native', lambda m,p:m)
    logits=torch.tensor([[-100.],[-20.],[-16.],[0.],[16.],[20.],[100.]])
    baseline=logits.sigmoid()
    target=(logits<0).float()
    op=torch.eye(len(logits)).to_sparse().coalesce()
    e=dict(view=0, op=op, adjoint=op, columns=torch.tensor([0]),
           target=target, known=torch.ones_like(logits))
    denominator=torch.ones_like(logits)
    expected=baseline.clone()
    for step,pre_state,_,forward in extent_training_states(
            logits,[e],set(),denominator,denominator,steps=2):
        # Independent recurrence: residual uses deployed forward; prior uses
        # pre-readout correction. Overwriting the state with odds fails this.
        probability=odds_probability(logits,expected-baseline)
        gradient=(target-probability).clamp(-.25,.25)-.1*(expected-baseline)
        expected=(expected+.95*gradient/1.1).clamp(0,1)
        torch.testing.assert_close(pre_state,expected,rtol=0,atol=1e-7)
        torch.testing.assert_close(forward,odds_probability(logits,expected-baseline),rtol=0,atol=1e-7)
    class Model:
        latent=torch.zeros(7,1)
        def logits(self,q,modality,rows=None):return logits.clone()
    # Identity kernel isolates serialization/readout from deliberate ridge
    # smoothing, already covered by the serialized candidate-selection test.
    state=dict(anchors=torch.ones(1,1).half(),inverse=torch.ones(1,1),
               residual=(pre_state-baseline).half(),concentration=20,readout='odds')
    path=tmp_path/'field.pt'
    torch.save(dict(kernel_baseline='pipeline',source_kernel_extent=state),path)
    payload=torch.load(path,weights_only=False)
    cold=Model();module.attach_source_kernel_extent(cold,payload)
    deployed=cold.logits(torch.ones(1,1),'text').sigmoid()
    torch.testing.assert_close(deployed,forward,rtol=0,atol=3e-4)
    # Executable old-bug counterexample: serialize the deployed probability
    # difference, applying odds twice. It fails the same tolerance by >100x.
    payload['source_kernel_extent']['residual']=(forward-baseline).half()
    broken=Model();module.attach_source_kernel_extent(broken,payload)
    assert (broken.logits(torch.ones(1,1),'text').sigmoid()-forward).abs().max()>.03
    payload['source_kernel_extent']['residual'].zero_()
    zero=Model();module.attach_source_kernel_extent(zero,payload)
    assert torch.equal(zero.logits(torch.ones(1,1),'text').sigmoid(),baseline)
