"""Same corrected query and signed solver: legacy map versus rebuilt source map."""
import argparse
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--scene',choices=['fern','orchids','fortress'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
old=base/f'nvos_{a.scene}_query_ropefix_control_20260917/report.json'
new=base/'nvos_ropefix_pipeline_20260917'/a.scene/'evaluation'/f'{a.scene}_signed/report.json'
x=json.loads(old.read_text());y=json.loads(new.read_text())
for key in ['manifest_sha256','query_receipt_sha256','signed_policy','signed_refinement','target_frame','threshold','threshold_comparison','score_semantics','score_resize','grid_wh','scoring_grid_wh']:
    assert x[key]==y[key],key
assert x['sam_reference_complex_rope_preserved'] and y['sam_reference_complex_rope_preserved']
rows={}
for name,v in y['results'].items():
    u=x['results'][name]
    for report,path in [(x,old),(y,new)]:assert sha256_file(path.parent/(name+'.pt'))==report['results'][name]['prediction_sha256']
    rows[name]={m:{'old':u[m],'new':v[m],'delta':v[m]-u[m]} for m in ['iou','average_precision','positive_support_recall']}
    rows[name]['signed_constraints']=y['refinement_records'][name]
r=dict(scene=a.scene,old_report_sha256=sha256_file(old),new_report_sha256=sha256_file(new),same_corrected_query=True,same_signed_solver=True,source_inventory_and_all_dependent_training_rebuilt=True,geometry_bitwise_exact=x['geometry_sha256']==y['geometry_sha256'],geometry_retention_audit='paper/artifacts/sam_ropefix_geometry_retention_20260917.json' if a.scene in ('fern','orchids') else None,pure_single_variable_model_ablation=False,independent_final_test=False,implementation_sha256=sha256_file(__file__),results=rows)
a.output.write_text(json.dumps(r,indent=2)+'\n')
import shutil
shutil.copy2(__file__,a.output.with_suffix('.source.py'))
print({k:round(v['iou']['delta']*100,3) for k,v in rows.items()})
