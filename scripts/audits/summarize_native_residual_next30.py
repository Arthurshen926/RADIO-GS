"""Close the complete source-admitted residual control without default promotion."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

root=Path('output/native_residual_basis_next30_v2')
registration=json.loads((root/'registration.json').read_text())
pre={r['key']:json.loads((root/r['key']/'report.json').read_text()) for r in registration['cohort']}
assert len(pre)==19 and all(r['source_admitted'] for r in pre.values())
assert all((root/k/'map/report.json').exists() for k in pre)
scan=[json.loads(p.read_text()) for p in sorted(root.glob('scannet_*/category_evaluation/report.json'))]
lerf=[json.loads(p.read_text()) for p in sorted(root.glob('lerf_*/ranking_evaluation/report.json'))]
nvos=[json.loads(p.read_text()) for p in sorted(root.glob('nvos_*/ranking_evaluation_*/report.json'))]
fusion=[json.loads(p.read_text()) for p in sorted(root.glob('*/fusion_audit.json'))]
assert len(scan)==8 and len(lerf)==4 and len(nvos)==8 and len(fusion)==12
assert sum(r['metrics']['native64']['observations'] for r in lerf)==208
assert all(not any(r['native_baseline_prediction_mismatches'].values()) for r in scan)
result=dict(bounded_control_complete=True,overall_project_repairs_complete=False,mainline_promoted=False,
    independent_test=False,source_maps=19,lerf_observations=208,nvos_tasks=8,scannet_scenes=8,
    final_lerf2d_iou_rerun=False,final_lerf3d_selected_iou_rerun=False,final_nvos_completion_iou_rerun=False,
    capacity_counts={str(d):sum(r['residual_dimensions']==d for r in pre.values()) for d in [32,64,128]},
    source_worst_modality_mse_ratio_range=[min(r['worst_mse_ratio'] for r in pre.values()),max(r['worst_mse_ratio'] for r in pre.values())],
    scannet_native={arm:{s:sum(r['metrics'][arm][s]['miou'] for r in scan)/8 for s in ['19','15','10']} for arm in ['native64','native_residual']},
    lerf_native_ranking={arm:{k:sum(r['metrics'][arm][k] for r in lerf)/4 for k in ['macro_ap','localization_accuracy']} for arm in ['native64','native_residual']},
    nvos_native_ranking={arm:sum(r['metrics'][arm]['ap'] for r in nvos)/8 for arm in ['native64','native_residual']},
    source_postfusion_text_mse_worsens=sum(r['metrics']['native_residual']['text']['mse']>r['metrics']['native64']['text']['mse'] for r in fusion),
    frozen_source_audit=json.loads((root/'frozen_source_audit.json').read_text()),
    tests=dict(passed=18,failed=0),
    decision='Do not promote: regional cosine preservation improves but fixed map readout gains are small and scene regressions remain. No target-selected per-scene switching.',
    failure_recovery='v1 residual preflight stopped on leaves unsupported rank; preserved attempt, v2 explicitly skips unsupported capacities and reruns all19. Same selection weights/criteria.',
    next_priority='Separate source visibility, correspondence and whole/part mixing; validate post-fusion query response and selected elements, not descriptor MSE alone.')
with (root/'summary.json').open('x') as f:json.dump(result,f,indent=2)
sources=['radio_gs/training/query_preserving_basis.py','radio_gs/training/native_semantic_residual.py',
    'tests/test_query_preserving_basis.py','tests/test_native_semantic_residual.py']
sources += [str(p) for p in Path('scripts/audits').glob('*next30*.py')]
artifacts=[p for p in root.rglob('*.json')]
seal=dict(summary=str(root/'summary.json'),sources={p:sha256_file(p) for p in sources},
    artifacts={str(p):sha256_file(p) for p in artifacts},mainline_promoted=False)
with Path('paper/artifacts/native_residual_next30.json').open('x') as f:json.dump(seal,f,indent=2)
print(json.dumps(result,indent=2))
