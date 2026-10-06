"""Aggregate corrected source diagnostics; never read target evaluation artifacts."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


root = Path('output/matched_source_fusion_next31')
records = []
for path in sorted(root.glob('*/report.json')):
    result = json.loads(path.read_text())
    oldpath = Path('output/native_residual_basis_next30_v2')/result['key']/'fusion_audit.json'
    old = json.loads(oldpath.read_text())
    metrics = result['metrics']
    error = max(abs(old['metrics'][arm][mode][oldkey]-metrics['legacy_'+arm][mode][newkey])
        for arm in ('native64','native_residual') for mode in ('text','image')
        for oldkey,newkey in (('mse','mse'),('centered_mse','centered_mse'),
                             ('teacher_top1_agreement','teacher_top1_agreement')))
    assert error < 1e-6, (result['key'],error)
    records.append(dict(key=result['key'],report_sha256=sha256_file(path),
        legacy_report_sha256=sha256_file(oldpath),legacy_reproduction_max_error=error,
        validation_regions=sum(r['regions'] for r in result['observations']),
        matched_regions=sum(r['matched_render_supported'] for r in result['observations']),
        common_ladder_regions=sum(r['common_ladder_supported'] for r in result['observations']),
        exact_mask_duplicate_pairs=sum(r['exact_duplicate_pairs'] for r in result['overlap']),
        near_identical_descriptor_duplicate_pairs=sum(r['exact_mask_and_descriptor_pairs'] for r in result['overlap']),
        containment_pairs=sum(r['containment_pairs'] for r in result['overlap']),
        containment_cosine_below_0_8=sum(r['containment_cosine_below_0_8'] for r in result['overlap']),
        metrics=metrics))
assert len(records) == 12 and sum(r['key'].startswith('lerf_') for r in records) == 4
stages = list(records[0]['metrics'])
macro = {stage:{mode:{metric:sum(r['metrics'][stage][mode][metric] for r in records)/len(records)
    for metric in records[0]['metrics'][stage][mode]} for mode in ('text','image')} for stage in stages}
summary = dict(schema='matched_source_fusion_diagnostic.next31',maps=12,
    map_macro=macro,records=records,legacy_reproduced=True,source_only=True,
    no_target_metrics=True,teacher_crop_response_not_pixel_gt=True,
    whole_part_or_cross_view_gaps_not_pure_geometry_causal_estimates=True,
    previous_residual_text_mse_worsened_maps=sum(r['metrics']['legacy_native_residual']['text']['mse']>
        r['metrics']['legacy_native64']['text']['mse'] for r in records),
    corrected_residual_text_mse_worsened_maps=sum(r['metrics']['matched_render_native_residual']['text']['mse']>
        r['metrics']['matched_render_native64']['text']['mse'] for r in records),
    exact_mask_duplicate_pairs=sum(r['exact_mask_duplicate_pairs'] for r in records),
    near_identical_descriptor_duplicate_pairs=sum(r['near_identical_descriptor_duplicate_pairs'] for r in records),
    implementation_sha256=sha256_file(__file__))
with (root/'summary.json').open('x') as handle:
    json.dump(summary,handle,indent=2)
print({stage:macro[stage]['text']['teacher_top1_agreement'] for stage in stages})
