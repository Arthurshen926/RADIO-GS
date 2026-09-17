from radio_gs.training.scope_correspondence import partition_source_scopes,scope_pair_relation


def row(a,b,iou=.8,precision=.9,recall=.9):
    return dict(group=0,source_frame=a,target_frame=b,iou=iou,precision=precision,recall=recall,visible_source_pixels=40,visible_target_pixels=40)


def test_containment_unknown_and_contradictions_are_not_equal_scope():
    parent=row(1,2,.3,.3,.9);part=row(2,1,.3,.9,.3)
    assert scope_pair_relation(parent,part)=='source_contains_target'
    assert scope_pair_relation(part,parent)=='source_part_of_target'
    assert scope_pair_relation(parent,parent)=='unknown'
    assert scope_pair_relation(parent,None)=='unknown'
    assert partition_source_scopes(0,[1,2],[parent,part])['clusters']==[[1],[2]]


def test_complete_link_cannot_bridge_incompatible_endpoints():
    rows=[row(1,2),row(2,1),row(2,3),row(3,2),row(1,3,0,0,0),row(3,1,0,0,0)]
    result=partition_source_scopes(0,[1,2,3],rows)
    assert len(result['clusters'])==2
    assert not any(1 in c and 3 in c for c in result['clusters'])
    assert result==partition_source_scopes(0,[3,2,1],list(reversed(rows)))


def test_heldout_adversarial_relations_cannot_change_partition():
    rows=[row(1,2),row(2,1)]
    original=partition_source_scopes(0,[1,2],rows)
    extra=[row(1,99,0,0,0),row(99,2),row(2,99),row(99,1)]
    assert partition_source_scopes(0,[1,2],rows+extra)==original
    assert partition_source_scopes(0,[1,2],rows[:1])['clusters']==[[1],[2]]


def test_unknown_is_neutral_but_explicit_conflict_blocks_bridging():
    rows=[row(1,2),row(2,1),row(2,3),row(3,2)]
    result=partition_source_scopes(0,[1,2,3],rows,allow_unknown_bridges=True)
    assert result['clusters']==[[1,2,3]]
    assert any(r['source']==1 and r['target']==3 and r['relation']=='unknown' for r in result['relations'])
    rows += [row(1,3,0,0,0),row(3,1,0,0,0)]
    result=partition_source_scopes(0,[1,2,3],rows,allow_unknown_bridges=True)
    assert not any(1 in c and 3 in c for c in result['clusters'])
