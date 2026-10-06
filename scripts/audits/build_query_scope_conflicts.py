"""Expose incompatible source targets sharing an accepted fixed scope head.

These are automatic teacher constraints, not verified physical object IDs.
Missing detections remain unknown. No benchmark query/label is consulted.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    assert not args.output.exists()
    path = args.root / 'counterfactual_extent_benefit_admission/field.pt'
    assert sha256_file(path) == json.loads(path.with_name('report.json').read_text())['field_sha256']
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = ElementRetrievalField(**payload['architecture'])
    model.load_state_dict(payload['state_dict'])
    model.requires_grad_(False)
    ep = args.root / 'caption_region_evidence.pt'
    assert sha256_file(ep) == json.loads(ep.with_suffix('.json').read_text())['sha256']
    evidence = torch.load(ep, map_location='cpu', weights_only=False)
    assert evidence['source_only'] and not evidence['benchmark_data_used']
    assert evidence['geometry_binding'] == payload['geometry_binding']
    groups, accepted, _ = model.grounded_transfer.route(evidence['text_embeddings'].float())
    frame_masks = {}
    for index, record in enumerate(evidence['records']):
        mask = evidence['masks'][index].bool()
        if record['confidence'] < .7 or int(mask.sum()) < 20:
            continue
        targets = frame_masks.setdefault(record['frame'], {})
        for query in record['phrase_indices']:
            if accepted[query]:
                targets[query] = targets.get(query, torch.zeros_like(mask)) | mask
    rows = []
    tested = 0
    for frame, targets in sorted(frame_masks.items()):
        by_group = {}
        for query in targets:
            by_group.setdefault(int(groups[query]), []).append(query)
        for group, queries in by_group.items():
            for left, right in combinations(sorted(queries), 2):
                a, b = targets[left], targets[right]
                if torch.equal(a, b):
                    continue  # Exact aliases are not separate conflicting constraints.
                tested += 1
                intersection = int((a & b).sum())
                iou = intersection / max(1, int((a | b).sum()))
                if iou >= .1:
                    continue
                rows.append(dict(frame=frame, group=group, query_indices=[left, right],
                                 phrases=[evidence['phrases'][left], evidence['phrases'][right]],
                                 areas=[int(a.sum()), int(b.sum())], intersection=intersection,
                                 iou=iou, relation='incompatible_teacher_scopes',
                                 physical_identity='unknown'))
    report = dict(schema='source_query_fixed_scope_conflicts.v1', source_only=True,
                  benchmark_data_used=False, training=False,
                  field_sha256=sha256_file(path), evidence_sha256=sha256_file(ep),
                  implementation_sha256=sha256_file(__file__),
                  nonidentical_same_head_pairs=tested, incompatible_pairs=len(rows),
                  affected_groups=len({r['group'] for r in rows}),
                  affected_queries=len({q for r in rows for q in r['query_indices']}),
                  observations=rows,
                  limitation='Co-observed automatic teacher masks, not physical identity GT. '
                             'Shared accepted concept head does not alone prove final outputs '
                             'identical: query-dependent benefit admission may choose native output. '
                             'No absent-view negatives or training labels are synthesized.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print({k: report[k] for k in ['nonidentical_same_head_pairs', 'incompatible_pairs',
                                'affected_groups', 'affected_queries']})


if __name__ == '__main__':
    main()
