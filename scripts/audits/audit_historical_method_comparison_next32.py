"""Local, CPU-only historical comparability audit; no metric recomputation."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENES = ('figurines', 'ramen', 'teatime', 'waldo_kitchen')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources, historical_checks = {}, []

    def read(name):
        path = ROOT / name
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())

    def check(path, expected, role):
        path = ROOT / path
        actual = digest(path)
        sources[str(path)] = actual
        historical_checks.append(dict(path=str(path), expected_sha256=expected,
                                      actual_sha256=actual, match=actual == expected, role=role))

    retained2 = read('paper/artifacts/v4_repair_historical_control_2d_20260914.json')
    retained3 = read('paper/artifacts/v4_repair_historical_control_20260914.json')
    historical2 = read('paper/artifacts/lerf2d_current_field_boundary_readout_stage_a_20260810.json')
    historical_scan = read('paper/artifacts/scannet_canonical_mpr_v3_gaussian_semantic_result_authority_20260802.json')
    historical_nvos = read('paper/artifacts/nvos_hierarchical_trust_local_positive_full8_exact_results_20260805.json')
    nvos_prereg = read('paper/artifacts/nvos_hierarchical_trust_local_positive_full8_target_preregistration_20260805.json')
    check(nvos_prereg['frozen_dataset_authority']['manifest'],
          nvos_prereg['frozen_dataset_authority']['manifest_sha256'], 'historical_signed_NVOS_manifest')
    current_lerf = read('output/full_field_rebuild_next19/evaluation_summary.json')
    current_scan = read('output/scannet_selected_visibility_next31/summary.json')
    current_nvos = read('paper/artifacts/nvos_component_reference_next31.json')
    for key in ('current_field_exact_coarse', 'current_field_target_rgb_grabcut_v1', 'historical_feature_only_sam3'):
        record = historical2[key]['source']
        check(record['path'], record['sha256'], key)
    exact2 = read(historical2['current_field_exact_coarse']['source']['path'])
    for record in exact2['source_artifacts']:
        check(record['path'], record['sha256'], 'historical_51_74_renderer_receipt')
    for key in ('totality_freeze', 'evaluator_report'):
        record = historical_scan[key]
        check(record['path'], record['sha256'], 'historical_scannet_' + key)
    exact_nvos = historical_nvos['exact_scoring']
    check(exact_nvos['results'], exact_nvos['results_sha256'], 'historical_nvos_result')
    inventory = read('paper/artifacts/v4_review_historical_inventory_20260914.json')
    for row in inventory:
        check(row['path'], row['expected_sha256'], 'historical_inventory_' + row['key'])

    budgets = []
    for scene in SCENES:
        authority = read(f'paper/artifacts/lerf_{scene}_source32_exact_mpr_rgb_authority_20260817.json')
        teacher = read(f'output/full_field_rebuild_next19/{scene}/reconstruction/field/teachers.json')
        geometry = read(f'output/depth_opacity_repair_20260918/{scene}/geometry_report.json')
        for suffix in ('control_2d', 'control'):
            receipt = read(f'output/v4_repair_{suffix}_20260914/{scene}/replay_receipt.json')
            check(receipt['source_result'], receipt['source_sha256'], f'{suffix}_actual_replayed_result')
        budgets.append(dict(scene=scene,
            historical_proposal_source_views=authority['construction']['selection']['selected_count'],
            historical_exact_mpr_compatible_views=authority['construction']['exact_mpr_authority']['compatible_count'],
            historical_available_teacher_frames=authority['construction']['frame_manifest']['source_only_count'],
            historical_region_input_policy=authority['information_policy'],
            current_feature_training_views=len(teacher['train_frames']),
            current_reserved_source_views=len(teacher['validation_frames']),
            current_moge_mapping_views=len(geometry['mapping_frames']),
            current_geometry_elements=geometry['results']['independent_scale']['elements'],
            complete_old_geometry_training_holdout_proved=False))
    scan_budgets = []
    scan_registration = read('output/scannet_selected_visibility_next31/registration.json')
    for row in scan_registration['scenes']:
        scene = row['scene']
        old = historical_scan['scenes'][scene]
        receipt = read(old['semantic_score_cache_receipt']['path'])
        check(old['semantic_score_cache_receipt']['path'], old['semantic_score_cache_receipt']['sha256'],
              'historical_scannet_score_receipt')
        mpr = read(receipt['mpr_source']['path'] + '.json')
        teacher = read(str(Path(row['root']) / 'teachers.json'))
        scan_budgets.append(dict(scene=scene, historical_mpr_views=mpr['num_views'],
            historical_geometry_elements=old['num_gaussians'],
            current_training_views=len(teacher['train_frames']),
            current_reserved_source_views=len(teacher['validation_frames']),
            historical_geometry='RGB-3DGS 30000-iteration optimized Gaussian rows',
            current_geometry='provided unlabelled mesh converted to Gaussian carrier',
            equal_metric_support=False))

    code_sources = [
        'CONTEXT.md', 'README.md',
        'docs/adr/0006-repair-v4-contracts-and-restore-performance-control.md',
        'docs/adr/0007-semantic-field-primary-with-moge-geometry-prior.md',
        'docs/adr/0009-protect-native-semantics-during-structural-learning.md',
        'docs/experiments/2026-09-26-historical-method-comparison-next32.md',
        'docs/experiments/2026-09-14-v4-repair-and-control-recovery.md',
        'docs/experiments/2026-08-17-lerf2d-historical-feature-sam3-current-field-audit.md',
        'paper/artifacts/protocol_alignment_audit_20260711.md',
        'paper/artifacts/five_contract_destination_compliant_baseline_gap_matrix_20260813.json',
        'paper/artifacts/canonical_compact_capability_iteration_20260714.md',
        'paper/artifacts/historical_hybrid_operator_audit_20260917.json',
        'paper/artifacts/lerf_identity_extent_dual_posterior_full4_result_20260817.json',
        'paper/artifacts/ours_lerf2d_scalar_render_authority_20260802.json',
        'paper/artifacts/evaluation_protocol_freeze_20260801.yaml',
        'paper/artifacts/evaluation_protocol_registry_20260731.yaml',
        'radio_gs/scripts/eval_prerendered_lerf_features.py',
        'radio_gs/scripts/eval_ours_lerf2d_scalar_maps.py',
        'radio_gs/scripts/render_ours_lerf2d_scalar_maps.py',
        'scripts/audits/evaluate_full_field_rebuild_next19.py',
        'scripts/audits/rebuild_retained_lerf_queries.py',
        'radio_gs/scripts/build_lerf_sam_siglip_object_posterior_scores.py',
        'radio_gs/querying/sam_siglip_object_posterior.py',
        'radio_gs/querying/latent_proposal_posterior.py',
        'radio_gs/querying/nvos_local_positive_completion.py',
        'radio_gs/models/hybrid_gaussian.py',
        'radio_gs/scripts/train_feature_field.py',
        'radio_gs/field/canonical_gaussian_field.py',
        'radio_gs/field/basis_decoder.py',
        'radio_gs/interfaces/surface_region_summary.py',
        'scripts/audits/audit_historical_method_comparison_next32.py',
    ]
    for name in code_sources:
        sources[name] = digest(ROOT / name)
    historical_mismatches = [r for r in historical_checks if not r['match']]
    report = dict(schema='historical_method_comparison.next32', audit_complete=True,
        scoring_or_training_run=False, gpu_used=False, independent_test=False,
        historical_bindings_checked=len(historical_checks), historical_binding_errors=historical_mismatches,
        large_checkpoints_maps_and_source_rgb_rehashed=False,
        source_hashes=sources,
        exact_historical_rows=dict(
            retained_lerf2d_scene_macro=sum(r['scene_macro_miou'] for r in retained2['scenes'])/4,
            retained_lerf2d_query_micro=retained2['sample_mean_iou'],
            retained_selected_scene_macro=retained3['scene_mean_iou'],
            retained_selected_query_micro=retained3['sample_mean_iou'],
            surface_region_v2_lerf2d=historical2['current_field_exact_coarse']['scene_macro'],
            target_rgb_grabcut=historical2['current_field_target_rgb_grabcut_v1']['scene_macro'],
            legacy_hcd_posthoc=historical2['historical_feature_only_sam3']['scene_macro'],
            scannet_canonical=historical_scan['metrics'],
            nvos_signed_iou=exact_nvos['scene_macro_foreground_iou']),
        current_retained_rows=dict(lerf=current_lerf['macro'],
            scannet=current_scan['macro']['retained'],nvos=current_nvos['macro']['baseline']),
        lerf_budget_audit=budgets, scannet_budget_audit=scan_budgets,
        historical_occam_protocol=exact2['protocol_config'],
        current_lerf2d_protocol=dict(mask='source-calibrated alpha-normalized observed probability > .5',
            localization='unsmoothed probability peak inside GT bbox',multi_scale=False,
            current_vs_occam_metric_identity=False,source='scripts/audits/evaluate_full_field_rebuild_next19.py:114'),
        scoring_comparability=dict(
            lerf2d='Same208 observation cohort, different score/render/readout/localization contracts.',
            selected='Same public PNG IoU scorer; historical element threshold .6 and peak-component guard differ from current .5 and no guard; geometry differs.',
            scannet='Same VALA formulas; Gaussian support, pseudo-GT and volume measure change with geometry.',
            nvos='Same8 targets; historical signed vs current positive-only prompts, and local LUDVIG target-RGB rights differ.'),
        high_impact_transfer=[
            dict(candidate='Identity-preserving relative query scope supervision',
                 historical_reference='build_lerf_sam_siglip_object_posterior_scores.py:208-287',
                 prerequisite='Keep native semantic identity and test scope actual selection/query ordering jointly; no posthoc swap of localization metric.'),
            dict(candidate='Core/context and physical-scale evidence organization before query matching',
                 historical_reference='radio_gs/interfaces/surface_region_summary.py:511-669',
                 prerequisite='Re-train with current feature domain and source-heldout score/rank objective; no copied RADIO-only head or all-parent merging.'),
            dict(candidate='Primitive-local explicit/hash residual with affine teacher decoder',
                 historical_reference='radio_gs/field/canonical_gaussian_field.py:124-162',
                 prerequisite='Evaluate each primitive before render, protect native coordinates, bake one field and prove cold replay; no old screen-space decoder swap.'),
        ],
        limitations=['Historical result hashing is not full training reproduction.',
            'Current controls are not a jointly frozen unified method.',
            'No old result is a same-geometry same-budget ablation against the current field.',
            'No new SOTA claim or benchmark gain is established by this audit.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(complete=True, historical_bindings_checked=len(historical_checks),
        mismatches=historical_mismatches, sources=len(sources),
        historical_lerf2d=report['exact_historical_rows']['surface_region_v2_lerf2d'])))
    if historical_mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
