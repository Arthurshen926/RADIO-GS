"""Bind query outputs to one map without changing or selecting predictions."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    records = [torch.load(group['readout'], map_location='cpu', weights_only=False) for group in plan['groups']]
    first = records[0]
    for record in records[1:]:
        if record['semantic_geometry'] != first['semantic_geometry'] or record['selection_geometry'] != first['selection_geometry']:
            raise ValueError('Queries must refer to the same frozen semantic and selection geometry.')
    semantic_support = torch.cat([
        record['semantic_observed'].reshape(-1, 1).expand(-1, record['scores'].shape[1])
        for record in records
    ], dim=1)
    selection_support = torch.cat([
        record['selection_observed'].reshape(-1, 1).expand(-1, record['selection_scores'].shape[1])
        for record in records
    ], dim=1)
    shared_support = bool((semantic_support == semantic_support[:, :1]).all() and
                          (selection_support == selection_support[:, :1]).all())
    result = dict(
        texts=[text for record in records for text in record['texts']],
        queries=[query for group in plan['groups'] for query in group['queries']],
        scores=torch.cat([record['scores'] for record in records], dim=1),
        selection_scores=torch.cat([record['selection_scores'] for record in records], dim=1),
        semantic_geometry=first['semantic_geometry'], selection_geometry=first['selection_geometry'],
        semantic_observed_by_query=semantic_support,
        selection_observed_by_query=selection_support,
        readout_contract='semantic_probability_and_independent_selected_set' if shared_support else 'per_query_support_semantic_probability_and_independent_selected_set',
        query_intents=[group['kind'] for group in plan['groups'] for _ in group['queries']],
        source_access=plan['access'], generation_plan=args.plan,
    )
    if shared_support:
        result['semantic_observed'] = semantic_support[:, 0]
        result['selection_observed'] = selection_support[:, 0]
    # Distinct observed domains must not be unioned: doing so changes response
    # normalization. Frozen benchmark scorers consume each group's unchanged
    # readout; mixed-query consumers select the support column for that query.
    result['group_readouts'] = [group['readout'] for group in plan['groups']]
    torch.save(result, args.output)


if __name__ == '__main__':
    main()
