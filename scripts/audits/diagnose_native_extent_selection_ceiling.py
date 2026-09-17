"""GT-only diagnostic ceiling; never an executable selection policy or fitting input."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
result=dict(schema='native_extent_selection_ceiling.v1',diagnostic_only=True,benchmark_labels_used=True,deployable=False,scenes={})
for scene in ['figurines','ramen']:
 root=Path('output/pixel_evidence_rebuild_20260917')/scene
 files={'native':root/'native_text_after.json','extent':root/'text_evaluation.json','retained':root/'counterfactual_extent_benefit_admission_validation/text.json','semantic':root/'semantic_extent_benefit_admission_validation/text.json'}
 docs={k:json.loads(v.read_text()) for k,v in files.items()};protocols=[{k:v for k,v in d['protocol'].items() if k!='fields'} for d in docs.values()];assert all(p==protocols[0] for p in protocols)
 names=dict(native='full_source_query_threshold_control',extent='joint_supervised_image_residual',retained='counterfactual_extent_benefit_admission',semantic='semantic_extent_benefit_admission');runs={k:d['runs'][names[k]] for k,d in docs.items()}
 obs={k:{(r['frame'],r['category']):r for r in v['observations']} for k,v in runs.items()};assert all(set(o)==set(obs['native']) for o in obs.values());keys=list(obs['native']);n=len(keys)
 native=obs['native'];extent=obs['extent'];categories=sorted({k[1] for k in keys});bycat={}
 for c in categories:
  ck=[k for k in keys if k[1]==c];means={m:sum(obs[m][k]['iou'] for k in ck)/len(ck) for m in obs};bycat[c]=dict(means,oracle_route='native' if means['native']>means['extent'] else 'extent',query_fixed_ceiling=max(means['native'],means['extent']),observations=len(ck))
 query_ceiling=sum(v['query_fixed_ceiling']*v['observations'] for v in bycat.values())/n
 observation_ceiling=sum(max(native[k]['iou'],extent[k]['iou']) for k in keys)/n
 changes=[dict(category=c,before=v['retained'],after=v['semantic'],delta=v['semantic']-v['retained']) for c,v in bycat.items() if abs(v['semantic']-v['retained'])>1e-8]
 result['scenes'][scene]=dict(inputs={str(p):sha256_file(p) for p in files.values()},observations=n,metrics={m:{k:r[k] for k in ['observation_miou','category_macro_miou','average_precision_macro']} for m,r in runs.items()},query_fixed_binary_selection_ceiling=query_ceiling,per_observation_binary_selection_ceiling=observation_ceiling,remaining_query_selection_gap=query_ceiling-runs['retained']['observation_miou'],categories_both_routes_below_25_percent=[c for c,v in bycat.items() if v['query_fixed_ceiling']<.25],per_category=bycat,semantic_changes=changes)
 print(scene,'retained',runs['retained']['observation_miou'],'query oracle',query_ceiling,'observation oracle',observation_ceiling,'bothweak',result['scenes'][scene]['categories_both_routes_below_25_percent'])
result['interpretation']='Ceiling only for unchanged binary native/extent responses and fixed current thresholds. GT route choices are diagnostics, not trained policy, not a ceiling for new candidates, local mixing or improved geometry.'
out=Path('paper/artifacts/native_extent_selection_ceiling_20260917.json');out.write_text(json.dumps(result,indent=2)+'\n')
