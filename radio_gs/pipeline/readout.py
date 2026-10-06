"""Read one query without mixing its score units or observed domain with others."""
from pathlib import Path
import torch


def load_query_readout(path, query_id):
    """Return semantic values and, when independently inferred, selected members.

    Native image/point queries remain cosine rankings. They have no implicit
    probability threshold or selected set. Mixed source-assisted artifacts carry
    a separate observed domain for every query, preserving each renderer's mass
    normalization.
    """
    value = torch.load(Path(path), map_location='cpu', weights_only=False)
    queries = value.get('queries', [dict(id=text) for text in value['texts']])
    index = next(i for i, query in enumerate(queries) if query.get('id', query.get('text')) == query_id)
    result = dict(query=queries[index], query_index=index)
    if 'scores' in value:
        result['semantic_values'] = value['scores'][:, index]
        result['semantic_geometry'] = value['semantic_geometry']
        result['semantic_observed'] = (value['semantic_observed_by_query'][:, index]
                                       if 'semantic_observed_by_query' in value else value['semantic_observed'])
        result['semantic_units'] = 'category_logits' if 'category_logits' in value else 'probability'
    else:
        result['semantic_values'] = value['cosine'][:, index]
        result['semantic_geometry'] = value['geometry']
        result['semantic_observed'] = value['observed'].reshape(-1)
        result['semantic_units'] = ('native_scope_compatibility' if value.get('matching_function') == 'late_logmeanexp' else 'native_cosine')
        if index in value.get('text_query_indices', []):
            text_index = value['text_query_indices'].index(index)
            probabilities = value.get('semantic_probability', value.get('text_semantic_probability'))
            if probabilities is not None:
                result['semantic_values'] = probabilities[:, text_index]
                result['semantic_units'] = 'probability'
    if 'selection_scores' in value:
        observed = (value['selection_observed_by_query'][:, index]
                    if 'selection_observed_by_query' in value else value['selection_observed'])
        result.update(selection_scores=value['selection_scores'][:, index], selection_observed=observed,
                      selection_geometry=value['selection_geometry'],
                      selected_element_ids=torch.where((value['selection_scores'][:, index] > .5) & observed)[0])
    else:
        result['selected_element_ids'] = None
    return result
