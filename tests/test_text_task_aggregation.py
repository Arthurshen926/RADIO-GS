from radio_gs.training.text_task_aggregation import canonical_text_tasks


def test_same_text_and_target_is_one_query_with_union_of_existing_evidence():
    a={'split':'validation','modality':'text','text_index':2,'target_view':3,'query_region':0,'target_region':6,'positive_regions':[6],'negative_regions':[7]}
    b={**a,'query_region':4,'negative_regions':[8]}
    image={**a,'modality':'image'}
    result=canonical_text_tasks([(10,a),(11,b),(12,image)])
    assert len(result)==2
    assert result[0][1]['negative_regions']==[7,8]
    assert result[0][1]['source_episode_indices']==[10,11]
    assert result[0][1]['supporting_query_regions']==[0,4]
    assert a['negative_regions']==[7] and b['negative_regions']==[8]
    assert result[1]==(12,image)


def test_text_tasks_never_merge_across_split_or_target_view():
    a={'split':'train','modality':'text','text_index':2,'target_view':3,'query_region':0,'target_region':6,'negative_regions':[]}
    assert len(canonical_text_tasks([(0,a),(1,{**a,'split':'validation'}),(2,{**a,'target_view':4})]))==3
