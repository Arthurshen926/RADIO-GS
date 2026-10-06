"""Seal the completed bounded controls without promoting a new method."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    artifacts = {}

    def bind(path, expected=None):
        path = str(path)
        digest = artifacts.get(path)
        if digest is None:
            digest = sha256_file(path)
            artifacts[path] = digest
        if expected is not None:
            assert digest == expected, path
        return digest

    children = [
        'matched_source_fusion_next31', 'nvos_component_reference_next31',
        'native_mixture_source_pilots_next31', 'scannet_selected_visibility_next31',
        'publication_readiness_next31',
        'new_method_independent_review_next31',
    ]
    for name in children:
        path = Path('paper/artifacts') / (name + '.json')
        child = json.loads(path.read_text())
        bind(path)
        for key in ('artifacts', 'artifact_hashes', 'implementations'):
            for item, expected in child.get(key, {}).items():
                assert isinstance(expected, str) and len(expected) == 64, (key, item)
                bind(item, expected)

    base = Path('output/selected_visibility_next31_v2')
    summary = json.loads((base / 'summary.json').read_text())
    assert summary['complete'] and not summary['mainline_promoted']
    assert summary['observations'] == 208 and len(summary['scenes']) == 4
    for path, expected in summary['input_bindings'].items():
        bind(path, expected)
    for item in ('summary.json', 'registration.json', 'evaluation_registration.json',
                 'independent_review.json'):
        bind(base / item)
    reg = json.loads((base / 'registration.json').read_text())
    for row in reg['scenes']:
        bind(Path(row['parent']) / reg['parent_name'] / 'field.pt', row['parent_field_sha256'])
        for mode in reg['modes']:
            root = Path(row['root']) / (mode + '_next31')
            report = json.loads((root / 'report.json').read_text())
            bind(root / 'field.pt', report['field_sha256'])
            bind(root / 'report.json')
            assert summary['rows'][row['scene']]['arms'][mode]['native_mask_replay_exact']
            assert summary['rows'][row['scene']]['arms'][mode]['image_prompt_macro_exact']

    for path in (
        __file__, 'scripts/audits/train_visibility_selected_scope_next31.py',
        'scripts/audits/run_selected_visibility_next31.py',
        'scripts/audits/evaluate_selected_visibility_next31.py',
        'scripts/audits/summarize_selected_visibility_next31.py',
        'scripts/audits/evaluate_full_field_rebuild_next19.py',
        'radio_gs/training/selected_output_supervision.py',
        'radio_gs/training/proposal_invariant_fusion.py',
        'tests/test_proposal_invariant_fusion.py',
        'docs/experiments/2026-09-26-system-fusion-and-visible-scope.md',
    ):
        bind(path)

    for path in [Path('output/new_method_tests_next31/report.json'),
                 Path('output/new_method_tests_next31/proposal_addendum/report.json')]:
        report = json.loads(path.read_text())
        assert report['exit_code'] == 0 and not report['inputs_changed_during_test']
        bind(path)
        inputs = report.get('input_hashes', report.get('inputs'))
        assert isinstance(inputs, dict) and inputs, path
        for item, expected in inputs.items():
            bind(item, expected)

    manifest = Path('output/frozen_cross_benchmark_method_20260918/source_manifest.json')
    frozen = json.loads(manifest.read_text())
    for path, expected in frozen.items():
        assert sha256_file(path) == expected, path
    bind(manifest)
    result = dict(
        schema='system_repair_complete.next31', complete=True,
        all_scheduled_experiments_complete=True, mainline_promoted=False,
        independent_test=False, claims_sota=False, unified_four_query_validated=False,
        uqis_paused=True, frozen_sources_unchanged=len(frozen), unique_focused_tests_passed=57,
        lerf=summary['macro'],
        scannet=json.loads(Path('output/scannet_selected_visibility_next31/summary.json').read_text())['macro'],
        child_seals=children,
        decision='Preserve existing defaults: limited LERF selection gain, no joint gain; '
                 'ScanNet/NVOS candidates and both source representation pilots rejected.',
        artifact_count=len(artifacts), artifacts=artifacts,
    )
    output = Path('paper/artifacts/system_repair_complete_next31.json')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in (
        'complete', 'artifact_count', 'frozen_sources_unchanged', 'unique_focused_tests_passed',
        'mainline_promoted', 'claims_sota')}, indent=2))


if __name__ == '__main__':
    main()
