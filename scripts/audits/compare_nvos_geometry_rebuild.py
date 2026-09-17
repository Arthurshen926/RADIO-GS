"""Compare frozen NVOS recipes only after prediction under matching contracts."""
import argparse,json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--new-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();base=Path('/mnt/pool/sqy/results/RADIO-GS/output');rows=[];inputs={}
def read(path):
 inputs[str(path)]=sha256_file(path);return json.loads(path.read_text())
for signed in [False,True]:
 new=read(a.new_root/'evaluation'/('fern_signed' if signed else 'fern_image')/'report.json')
 for support in [False,True]:
  old=read(base/('nvos_fern_'+('signed_' if signed else '')+('measured_support512' if support else 'image_route')+'_20260917')/'report.json')
  for key in ['manifest_sha256','query_receipt_sha256','target_frame','threshold','threshold_comparison','score_semantics','score_resize','grid_wh','scoring_grid_wh','signed_refinement']:
   if key=='signed_refinement':
    assert old.get(key,False)==new[key] and bool(new[key])==signed,('evaluation contract changed',key)
    if key not in old:assert not old.get('refinement_records') and not signed
   else:assert old[key]==new[key],('evaluation contract changed',key)
  for report in [old,new]:
   assert not report['target_rgb_opened'] and report['target_gt_used_after_prediction_only'] and report['training_feature_reads_forbidden']
  for residual in [False,True]:
   run=('image_supervised_native_residual' if residual else 'nvos96_protected_imagecal_balanced')+('_measured_support512' if support else '')
   before,after=old['results'][run],new['results'][run]
   rows.append(dict(signed=signed,support512=support,image_residual=residual,run=run,old=before,new=after,iou_delta=after['iou']-before['iou'],ap_delta=after['average_precision']-before['average_precision'],old_field_sha256=old['fields'][run],new_field_sha256=new['fields'][run]))
result=dict(schema='nvos_geometry_rebuild_comparison.v1',scene='fern',rows=rows,inputs=inputs,all_variants_reported=True,posthoc_best_variant_selected=False,full_caption_method=False,development_evaluation=True,limitations=['New geometry requires rebuilding source assets and training; this is a full recipe rebuild, not an isolated geometry-only causal estimate','Source inventory and task counts can differ; source training coverage policy stays two balanced passes','Sparse geometry improvements do not imply dense boundary or retrieval improvements','Previously inspected scene, not independent final test'],implementation_sha256=sha256_file(__file__))
a.output.write_text(json.dumps(result,indent=2)+'\n');print([(r['signed'],r['support512'],r['image_residual'],r['iou_delta']) for r in rows])
