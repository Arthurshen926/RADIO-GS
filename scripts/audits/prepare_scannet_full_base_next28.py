"""Reuse bound source RGB/SAM assets while rebuilding the complete source field."""
import argparse
import json
from pathlib import Path
import shutil

from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    scene = args.scene
    root = (Path('output/scannet_coverage_native_probe_20260919') if scene in ('scene0000_00', 'scene0062_00')
            else Path('output/scannet_coverage_native_next28')) / scene
    native = json.loads((root/'execution.json').read_text())
    if not native['complete']:
        raise RuntimeError('Native source stage is incomplete')
    source = root/'base'
    target = root/'base_full_next28'
    target.mkdir(exist_ok=False)
    (target/'source_sam').mkdir()
    (target/'sam_sources').mkdir()
    bindings = {}
    for name in ('source_plan.json', 'source_sam_execution.json'):
        src = source/name
        shutil.copy2(src, target/name)
        bindings[name] = sha256_file(src)
        assert sha256_file(target/name) == bindings[name]
    src = source/'source_sam/manifest.json'
    shutil.copy2(src, target/'source_sam/manifest.json')
    bindings['source_sam/manifest.json'] = sha256_file(src)
    for item in (source/'sam_sources').iterdir():
        if item.is_file():
            shutil.copy2(item, target/'sam_sources'/item.name)
            bindings['sam_sources/'+item.name] = sha256_file(item)
    plan = json.loads((target/'source_plan.json').read_text())
    visibility = json.loads((Path('output/scannet_visibility_audit_next28')/scene/'report.json').read_text())
    if set(plan['train_frames']) != set(visibility['equal_budget_geometry_selected_frames']):
        raise ValueError('Full source frames differ from audited source-only selection')
    if set(plan['validation_frames']) != set(visibility['validation_views']):
        raise ValueError('Full validation frames differ')
    receipt = dict(scene=scene, native_execution_sha256=sha256_file(root/'execution.json'),
                   same_source_frames=True, source_sam_assets_reused=True,
                   source_sam_manifest_contains_absolute_original_paths=True,
                   complete_field_not_reused=True, bindings=bindings,
                   geometry_report_sha256=sha256_file(root/'geometry_report.json'),
                   implementation_sha256=sha256_file(__file__))
    (target/'reuse_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
