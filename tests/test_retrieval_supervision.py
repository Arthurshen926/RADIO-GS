import torch
import pytest
from radio_gs.training.retrieval_supervision import membership_loss,OptionalTeacherHeads,boundary_loss

def test_unknown_mask_is_not_negative():
 p=torch.tensor([.8,.1,.9],requires_grad=True);y=torch.tensor([1.,0.,float('nan')]);known=torch.tensor([True,True,False])
 membership_loss(p,y,known).backward()
 assert p.grad[2]==0 and p.grad[0]<0 and p.grad[1]>0

def test_radio_can_be_absent_while_other_teachers_train():
 bank=OptionalTeacherHeads(4,{'dino':6,'siglip':8},{'radio':0,'dino':.1,'siglip':.2,'sam':.1})
 x=torch.randn(5,4,requires_grad=True)
 loss=bank.feature_loss('radio',x,None)+bank.feature_loss('dino',x,torch.randn(5,6))+bank.feature_loss('siglip',x,torch.randn(5,8))
 loss.backward()
 assert x.grad.abs().sum()>0 and 'radio' not in bank.heads
 assert all(p.grad is not None for p in bank.parameters())
 with pytest.raises(ValueError):bank.feature_loss('dino',x,None)

def test_boundary_unknown_edges_do_not_create_labels():
 p=torch.rand(4,requires_grad=True);known=torch.zeros(4,dtype=torch.bool)
 boundary_loss(p,torch.zeros(4),known,2,2).backward()
 assert torch.count_nonzero(p.grad)==0


def test_relation_loss_is_basis_invariant_and_detaches_teacher():
    from radio_gs.training.retrieval_supervision import relation_loss
    z = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
    teacher = torch.tensor([[0., 1.], [-1., 0.], [-1., 1.]], requires_grad=True)
    loss = relation_loss(z, teacher)
    assert loss.item() < 1e-10
    loss.backward()
    assert teacher.grad is None


def test_sam_contour_loss_has_no_unobserved_edge_gradient():
    from radio_gs.training.retrieval_supervision import sam_contour_loss
    latent = torch.randn(4, 3, requires_grad=True)
    loss = sam_contour_loss(latent, torch.tensor([1, 0, 0, 0]),
                            torch.tensor([1, 0, 0, 0], dtype=torch.bool), 2, 2)
    loss.backward()
    assert loss.item() == 0
    assert not latent.grad.any()


def test_sam_contour_prefers_feature_change_at_contour():
    from radio_gs.training.retrieval_supervision import sam_contour_loss
    smooth = torch.tensor([[1., 0.], [1., 0.]])
    edge = torch.eye(2)
    mask = torch.tensor([True, False]);valid = torch.ones(2, dtype=torch.bool)
    assert sam_contour_loss(edge, mask, valid, 1, 2) < sam_contour_loss(smooth, mask, valid, 1, 2)


def test_text_union_conflicts_and_unobserved_regions():
    from radio_gs.training.retrieval_supervision import episode_membership_targets
    masks=torch.tensor([[1,0,0,0],[0,1,1,0],[0,0,1,1]],dtype=torch.bool)
    episode={'target_region':0,'positive_regions':[0,1],'negative_regions':[2]}
    target,known,row_target,row_known=episode_membership_targets(
        episode,masks,masks.float(),torch.tensor([1,1,1,0],dtype=torch.bool),torch.ones(4))
    assert target.tolist()==[1,1,1,0]
    assert known.tolist()==[True,True,False,True]
    assert row_known.tolist()==[True,True,False,False]


def test_absence_is_not_negative():
    from radio_gs.training.retrieval_supervision import episode_membership_targets
    masks=torch.tensor([[True,False]])
    _,known,_,row_known=episode_membership_targets(
        {'target_region':0,'negative_regions':[]},masks,masks.float(),torch.ones(2,dtype=torch.bool),torch.ones(2))
    assert known.tolist()==row_known.tolist()==[True,False]


def test_soft_query_relations_preserve_uncertainty_and_detach_teacher():
    from radio_gs.training.retrieval_supervision import soft_query_relation_loss
    teacher=torch.tensor([[.1,.1,.2],[.3,.1,.1]],requires_grad=True)
    student=teacher.detach().clone().requires_grad_()
    loss=soft_query_relation_loss(student,teacher)
    assert abs(loss.item())<1e-6
    loss.backward()
    assert teacher.grad is None
    wrong=teacher.detach().flip(1).requires_grad_()
    loss=soft_query_relation_loss(wrong,teacher)
    assert loss.item()>.1
    loss.backward()
    assert wrong.grad.abs().sum()>0


def test_source_coverage_schedule_excludes_heldout_and_covers_unmatched():
    from radio_gs.training.retrieval_supervision import source_region_schedule
    views=torch.tensor([0,0,1,1,1,2,3])
    assert source_region_schedule(views,[0,1])==[0,2,1,3,4]


def test_split_validator_rejects_query_side_leakage():
    import pytest
    from radio_gs.training.retrieval_supervision import validate_episode_split
    episode={'split':'train','query_region':1,'target_region':0,'target_view':0,'negative_regions':[]}
    with pytest.raises(ValueError,match='leaks across split'):
        validate_episode_split([episode],torch.tensor([0,1]),[11,88],[11],[88])


def test_split_validator_rejects_wrong_target_region_view():
    import pytest
    from radio_gs.training.retrieval_supervision import validate_episode_split
    episode={'split':'train','query_region':0,'target_region':0,'target_view':0,'negative_regions':[1]}
    with pytest.raises(ValueError,match='different view'):
        validate_episode_split([episode],torch.tensor([0,1,2]),[11,21,88],[11,21],[88])


def test_fixed_step_budget_can_skip_many_episodes_in_a_new_scene():
    from radio_gs.training.retrieval_supervision import balanced_episode_coverage
    coverage=balanced_episode_coverage({'text':34,'image':84,'prompt':84},180)
    assert coverage['visited']=={'text':34,'image':60,'prompt':60}
    assert not coverage['complete']
    assert coverage['minimum_steps_for_full_coverage']==252
    assert balanced_episode_coverage({'text':34,'image':84,'prompt':84},252)['complete']


def test_episode_coverage_reports_training_exposure_not_just_visits():
    from radio_gs.training.retrieval_supervision import balanced_episode_coverage
    old = balanced_episode_coverage({'text':34,'image':84,'prompt':84},408)
    new = balanced_episode_coverage({'text':50,'image':136,'prompt':136},408)
    assert old['complete'] and new['complete']
    assert old['episode_repetitions']['image']['maximum'] == 2
    assert new['episode_repetitions']['image'] == {'minimum':1,'maximum':1,'mean':1.0}
    assert old['updates']['image'] == new['updates']['image'] == 136


def test_query_neighborhood_is_rotation_invariant_and_detaches_native():
    from radio_gs.training.retrieval_supervision import query_neighborhood_loss
    native = torch.eye(4, requires_grad=True)
    projected = native.detach().roll(1, 1).requires_grad_()
    loss = query_neighborhood_loss(projected, native)
    assert abs(float(loss)) < 1e-6
    loss.backward()
    assert native.grad is None
    wrong = torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,-1.]], requires_grad=True)
    loss = query_neighborhood_loss(wrong,native)
    loss.backward()
    assert loss > 0 and wrong.grad.norm() > 0


def test_epoch_shuffle_covers_tasks_without_changing_global_rng():
    from radio_gs.training.retrieval_supervision import episode_order
    state = torch.random.get_rng_state()
    order = episode_order(20,0,42,True)
    assert sorted(order) == list(range(20))
    assert order == episode_order(20,0,42,True)
    assert order != episode_order(20,1,42,True)
    assert torch.equal(state,torch.random.get_rng_state())
    assert episode_order(20,1,42,False) == list(range(20))


def test_unit_teacher_aggregation_matches_query_cosines_and_ignores_latent_scale():
    from radio_gs.training.retrieval_supervision import teacher_element_features
    z=torch.tensor([[1.,0.],[0.,3.],[9.,9.]],requires_grad=True)
    known=torch.tensor([True,True,False])
    weights=torch.tensor([[.25,.75,0.]])
    directions=teacher_element_features(z,known,True)
    rendered=weights@directions
    scaled=teacher_element_features(z*torch.tensor([[7.],[.2],[100.]]),known,True)
    assert torch.allclose(rendered,weights@scaled)
    query=torch.tensor([[.6,.8]])
    assert torch.allclose(rendered@query.T,weights@(directions@query.T))
    assert not torch.allclose(weights@teacher_element_features(z,known,False),weights@teacher_element_features(z*2,known,False))
    rendered.sum().backward()
    assert z.grad[2].eq(0).all()


def test_symmetric_matching_preserves_query_identity_with_one_region():
    from radio_gs.training.retrieval_supervision import symmetric_query_region_relation_loss
    teacher=torch.tensor([[.9],[.1],[.2]],requires_grad=True)
    student=torch.tensor([[.1],[.9],[.2]],requires_grad=True)
    loss=symmetric_query_region_relation_loss(student,teacher)
    assert loss>0
    loss.backward()
    assert student.grad.abs().sum()>0
    assert teacher.grad is None
    assert symmetric_query_region_relation_loss(teacher.detach(),teacher.detach()).abs()<1e-6
