"""Verify and seal the registered-prompt development candidate and all controls."""
import json
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    archive = base / 'scale_affinity_sources_20260916'
    archive.mkdir(exist_ok=True)
    current = {}
    for path in map(Path, [__file__, 'radio_gs/models/element_retrieval_field.py',
                          'radio_gs/models/scale_affinity_readout.py',
                          'scripts/audits/train_scale_affinity.py',
                          'scripts/audits/export_native_seed_affinity.py',
                          'scripts/audits/plot_registered_structure_queries.py',
                          'scripts/audits/evaluate_real_reference_queries.py',
                          'scripts/audits/evaluate_frozen_field_lerf_labels.py']):
        digest = sha256_file(path)
        snapshot = archive / (digest + '.py')
        if not snapshot.exists():
            snapshot.write_bytes(path.read_bytes())
        assert sha256_file(snapshot) == digest
        current[str(path)] = digest
    parent_name = 'preserved96_protected_imagecal_balanced'
    candidate_name = 'scale_affinity_v1_direct_control'
    scenes = {}
    artifacts = {}
    for scene in ['figurines', 'ramen']:
        root = base / 'preserved_source_inventory_20260916' / scene
        reports = {}
        for name in [parent_name, candidate_name, 'scale_affinity_v1',
                     'native_seed_affinity_v1', 'registered_full_field_control']:
            report = json.loads((root / name / 'report.json').read_text())
            assert sha256_file(root / name / 'field.pt') == report['field_sha256']
            reports[name] = report
            for digest in report.get('implementation_sources', {}).values():
                # Older parent belongs to a separate sealed training run.
                if name != parent_name:
                    assert sha256_file(archive / (digest + '.py')) == digest
            artifacts[str(root / name / 'report.json')] = sha256_file(root / name / 'report.json')
        parent = torch.load(root / parent_name / 'field.pt', weights_only=False, map_location='cpu')
        candidate = torch.load(root / candidate_name / 'field.pt', weights_only=False, map_location='cpu')
        for key, value in parent['state_dict'].items():
            assert torch.equal(value, candidate['state_dict'][key]), key
        extra = set(candidate['state_dict']) - set(parent['state_dict'])
        assert extra and all(key.startswith('affinity.') for key in extra)
        report = reports[candidate_name]
        assert report['parent_field_sha256'] == reports[parent_name]['field_sha256']
        assert report['default_scale_index'] == max(range(len(report['validation_before'])), key=report['validation_before'].__getitem__)
        assert not report['structural_training'] and not report['benchmark_data_used']
        assert not set(report['source_training_regions']) & set(report['source_validation_regions'])
        evaluations = {}
        for suffix in ['scale_affinity_reference', 'native_seed_affinity_text',
                       'native_seed_affinity_reference', 'registered_full_reference',
                       'registered_seed_validity', 'registered_structure_comparison']:
            extension = 'png' if suffix.endswith('comparison') else 'json'
            path = Path(f'paper/artifacts/{scene}_{suffix}_20260916.{extension}')
            artifacts[str(path)] = sha256_file(path)
            if extension == 'json':
                evaluations[suffix] = json.loads(path.read_text())
        evaluation = evaluations['scale_affinity_reference']
        assert evaluation['protocol']['implementation_sha256'] == current['scripts/audits/evaluate_real_reference_queries.py']
        assert not evaluation['protocol']['target_rgb_encoded_for_its_own_query']
        runs = evaluation['runs']
        for name, run in runs.items():
            assert run['field_sha256'] == reports[name]['field_sha256']
        before, after = runs[parent_name], runs[candidate_name]
        assert len(before['observations']) == len(after['observations'])
        for old, new in zip(before['observations'], after['observations']):
            for key in ['reference', 'target_frame', 'category', 'text', 'image']:
                assert old[key] == new[key], (scene, key)
        for metric in ['iou', 'average_precision']:
            assert after['macro']['prompt'][metric] > before['macro']['prompt'][metric]
        seeds = evaluations['registered_seed_validity']
        assert len(seeds) == len(evaluation['references'])
        assert all(r['supplied'] == r['snapped'] and r['positive_known'] and r['negative_known'] for r in seeds)
        scenes[scene] = dict(candidate=str(root / candidate_name / 'field.pt'),
                             field_sha256=report['field_sha256'],
                             all_parent_tensors_equal=True,
                             text_image_observation_metrics_equal=True,
                             source_selected_scale_index=report['default_scale_index'],
                             observations=len(after['observations']),
                             references=len(seeds),
                             before=before['macro'], after=after['macro'])
    result = dict(schema='registered_structural_query_repair.v1', scenes=scenes,
                  status='registered prompt development candidate; not joint benchmark promotion',
                  rejected_controls=['scale_affinity_v1', 'native_seed_affinity_v1', 'registered_full_field_control'],
                  evaluation_scope='LERF derived two-point registered category-union development; not official point benchmark or independent test',
                  implementation_sources=current, artifacts=artifacts)
    destination = Path('paper/artifacts/registered_structural_query_repair_20260916.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
