"""Verify complete new-geometry reconstruction and matched development scores."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    out = Path('paper/artifacts/depth_opacity_semantic_result_20260918.json')
    assert not out.exists()
    results, bindings = {}, {}
    fixroot = Path('output/depth_opacity_pipeline_20260918/geometry_resolution_fix')
    fix = json.loads((fixroot/'receipt.json').read_text())
    for path, digest in fix['files'].items():
        assert sha256_file(fixroot/Path(path).name) == digest
    bindings[str(fixroot/'receipt.json')] = sha256_file(fixroot/'receipt.json')
    run = 'counterfactual_extent_benefit_admission'
    for scene in ['figurines', 'ramen']:
        root = Path('output/depth_opacity_pipeline_20260918')/scene
        record = json.loads((root/'execution.json').read_text()); assert record['complete']
        # Verify inner source-stage receipts, not only the wrapper's digest.
        for subdir, complete_key in [('base', 'base_finished'), ('reconstruction', 'finished')]:
            subroot = root/subdir
            subrecord = json.loads((subroot/'execution.json').read_text())
            assert subrecord[complete_key]
            for stage in subrecord['stages']:
                assert stage['status'] == 'complete'
                for path, digest in stage['outputs'].items(): assert sha256_file(path) == digest
                for path, digest in stage['implementations'].items():
                    assert sha256_file(subroot/'sources'/stage['name']/Path(path).name) == digest
        for stage in record['stages']:
            assert stage['status'] == 'complete'
            for path, digest in stage['outputs'].items(): assert sha256_file(path) == digest
            for path, digest in stage['implementations'].items(): assert sha256_file(root/'sources'/stage['name']/Path(path).name) == digest
        geometry_report = Path('output/depth_opacity_repair_20260918')/scene/'geometry_report.json'
        assert sha256_file(geometry_report) == record['geometry_report_sha256']
        geometry = json.loads(geometry_report.read_text())['results']['independent_scale']
        assert sha256_file(geometry['geometry']) == geometry['geometry_sha256']
        field = root/'reconstruction/field'
        authority = json.loads((field/'responsibility.json').read_text())
        assert authority['metadata']['frozen_geometry_sha256'] == geometry['geometry_sha256']
        assert authority['metadata']['frozen_geometry_report_sha256'] == sha256_file(geometry_report)
        row = {}
        for task in ['text', 'reference']:
            oldpath = Path('output/v2_scope_rebuild_20260918')/scene/(task+'_online.json')
            path = root/(task+'_online.json'); replay_path = root/(task+'_replay.json')
            old, new, replay = [json.loads(x.read_text()) for x in [oldpath, path, replay_path]]
            assert new['runs'] == replay['runs'], 'Online/replay prediction differs'
            assert old['protocol']['labels'] == new['protocol']['labels']
            a, b = old['runs'][run], new['runs'][run]
            if task == 'text':
                assert old['protocol']['width'] == new['protocol']['width'] and old['protocol']['threshold'] == new['protocol']['threshold']
                assert [(r['frame'], r['category']) for r in a['observations']] == [(r['frame'], r['category']) for r in b['observations']]
                row[task] = dict(before={k:a[k] for k in ['observation_miou','average_precision_macro','positive_support_recall_macro']}, after={k:b[k] for k in ['observation_miou','average_precision_macro','positive_support_recall_macro']}, categories_before=a['per_category_iou'], categories_after=b['per_category_iou'], tasks=b['tasks'])
            else:
                assert [(r['reference_frame'], r['target_frame'], r['category']) for r in a['observations']] == [(r['reference_frame'], r['target_frame'], r['category']) for r in b['observations']]
                row[task] = dict(before=a['macro'], after=b['macro'], tasks_per_modality=b['tasks_per_modality'], unavailable_prompts_before=a['unavailable_prompt_tasks'], unavailable_prompts_after=b['unavailable_prompt_tasks'])
            for f in [oldpath, path, replay_path]: bindings[str(f)] = sha256_file(f)
        for path in [geometry_report, root/'execution.json', field/run/'field.pt', field/run/'report.json', field/'responsibility.json']:
            bindings[str(path)] = sha256_file(path)
        selected = root/'selected_output/report.json'
        if selected.exists():
            d = json.loads(selected.read_text()); row['selected_only'] = dict(mean_iou=d['mean_iou'], tasks=d['tasks'], official_gt_equivalence_verified=False); bindings[str(selected)] = sha256_file(selected)
        results[scene] = row
    out.write_text(json.dumps(dict(schema='depth_opacity_semantic_result.v1', independent_final_test=False, new_geometry_all_dependent_assets_rebuilt=True, original_v2_control_preserved=True, online_replay_exact=True, implementation_sha256=sha256_file(__file__), results=results, bindings=bindings), indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
