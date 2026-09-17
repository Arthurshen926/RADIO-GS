"""A text query has no source image identity: aggregate its label evidence."""


def canonical_text_tasks(indexed_episodes):
    result=[]
    groups={}
    for index,episode in indexed_episodes:
        if episode['modality']!='text':
            result.append((index,episode))
            continue
        key=(episode['split'],episode['text_index'],episode['target_view'])
        if key not in groups:
            merged=dict(episode)
            merged['positive_regions']=list(episode.get('positive_regions',[episode['target_region']]))
            merged['negative_regions']=list(episode['negative_regions'])
            merged['source_episode_indices']=[]
            merged['supporting_query_regions']=[]
            groups[key]=merged
            result.append((index,merged))
        merged=groups[key]
        merged['positive_regions']=sorted(set(merged['positive_regions'])|set(episode.get('positive_regions',[episode['target_region']])))
        merged['negative_regions']=sorted(set(merged['negative_regions'])|set(episode['negative_regions']))
        merged['source_episode_indices'].append(index)
        merged['supporting_query_regions']=sorted(set(merged['supporting_query_regions'])|{episode['query_region']})
    return result
