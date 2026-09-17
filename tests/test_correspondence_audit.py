from radio_gs.training.correspondence_audit import audit_correspondence_graph


def episode(left,right,split='train'):
    return {'query_region':left,'target_region':right,'split':split,'modality':'image'}


def test_pairwise_matches_can_form_inconsistent_track():
    result=audit_correspondence_graph([episode(0,1),episode(1,2),episode(2,3)],[0,1,2,0])
    assert result['train']['unique_pairs']==3
    assert result['train']['inconsistent_components']==1
    assert result['train']['components'][0]['same_view_conflicts']=={'0':[0,3]}


def test_duplicate_directions_and_separate_splits_do_not_merge():
    result=audit_correspondence_graph([episode(0,1),episode(1,0),episode(1,2,'validation')],[0,1,0])
    assert result['train']['unique_pairs']==1
    assert result['train']['inconsistent_components']==0
    assert result['validation']['inconsistent_components']==0


def test_conflicts_are_query_specific_and_never_use_validation_labels():
    import torch
    from radio_gs.training.correspondence_audit import training_query_conflicts
    probability=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    def e(target,neg,split='train',word=0):
        return {'split':split,'modality':'text','query_region':0,'text_index':word,
                'target_region':target,'negative_regions':neg}
    episodes=[e(0,[1]),e(1,[0]),e(2,[0],'validation'),e(2,[0],word=1)]
    conflicts=training_query_conflicts(episodes,probability)
    assert list(conflicts)==[('text',0)]
    assert conflicts[('text',0)].tolist()==[True,True,False]
