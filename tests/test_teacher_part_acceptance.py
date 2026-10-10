import torch
from scripts.research.teacher_part_acceptance import select_parts


def test_disconnected_teacher_union_selects_both_parts_and_abstains_on_empty():
    masks=torch.zeros(3,4,6,dtype=torch.bool)
    masks[0,:,0:2]=True;masks[1,:,4:6]=True;masks[2,:,2:4]=True
    instance=dict(scores=torch.ones(2),masks=masks[:2].float())
    p,d=select_parts(masks,instance)
    assert d['source_feasible_iou']==1 and set(d['selected'])=={0,1}
    assert torch.equal(p,torch.tensor([1.,1.,.5]))
    empty,details=select_parts(masks,dict(scores=torch.empty(0),masks=torch.empty(0,4,6)))
    assert bool((empty==.5).all()) and details['selected']==[] and not details['teacher_present']


def test_duplicate_candidates_do_not_count_twice():
    masks=torch.ones(3,2,2,dtype=torch.bool)
    p,d=select_parts(masks,dict(scores=torch.ones(1),masks=masks[:1].float()))
    assert d['selected']==[0] and p.tolist()==[1.,.5,.5]
