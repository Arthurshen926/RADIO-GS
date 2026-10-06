"""Seal complete bounded prompt-transfer cohort; no prediction or policy selection."""
import json
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/nvos_reference_completion_next19');execution=json.loads((root/'execution.json').read_text());assert execution['complete']
registration=json.loads((root/'registration.json').read_text());assert execution['script_sha256']==sha256_file('scripts/audits/nvos_reference_completion_next19_evaluate.py')
variants=registration['variants'];rows={};bindings={};tasks=registration['tasks'];assert len(tasks)==8
manifest=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text())
for task in tasks:
 report_path=root/task/'report.json';report=json.loads(report_path.read_text());assert report['baseline_mask_replay_exact'];assert not report['negative_scribble_opened'] and not report['target_rgb_opened'] and report['target_gt_used_after_prediction_only']
 old_report=json.loads((Path('output/nvos_system_refinement_20260919_v3')/task/'report.json').read_text())
 for metric in ['iou','pixel_accuracy']:assert report['results']['baseline'][metric]==old_report['results']['spatial_component'][metric]
 parent=Path('output/nvos_ropefix_pipeline_20260917')/('horns_center' if task=='horns_left' else task)
 field=parent/'field/image_supervised_native_residual_measured_support512/field.pt'
 assert sha256_file(field)==report['fields']['baseline']
 payload=torch.load(field,map_location='cpu',weights_only=False)
 shape=list(payload['state_dict']['latent'].shape);native_shape=list(payload['state_dict']['native_decoder.weight'].shape)
 rows[task]=dict(metrics=report['results'],reference_constraints=report['refinement_records'],field_shape=shape,native_decoder_shape=native_shape,report_sha256=sha256_file(report_path),baseline_response_max_abs_error=report['baseline_response_max_abs_error'])
 scene=next(s for s in manifest['scenes'] if s['scene_id']==task)
 target=next(f for f in scene['frames'] if f['frame_id'] in scene['evaluation_frame_ids'])
 assert sha256_file(target['gt_mask_path'])==target['ground_truth_sha256']
 truth=np.asarray(Image.open(target['gt_mask_path']));truth=np.any(truth>127,axis=2) if truth.ndim==3 else truth>127
 rows[task]['error_diagnostic']={}
 for variant in variants:
  response=torch.load(root/task/(variant+'.pt'),map_location='cpu',weights_only=False)['response'].numpy()
  selected=cv2.resize(response,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)>=.5
  tp=int((selected&truth).sum());fp=int((selected&~truth).sum());fn=int((~selected&truth).sum())
  rows[task]['error_diagnostic'][variant]=dict(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),tp=tp,fp=fp,fn=fn)
  assert abs(tp/max(1,tp+fp+fn)-report['results'][variant]['iou'])<1e-7
  pred=root/task/(variant+'.pt');assert sha256_file(pred)==report['results'][variant]['prediction_sha256'];bindings[str(pred)]=sha256_file(pred)
 for variant in variants:
  ep=root/task/(variant+'_elements.pt');element=torch.load(ep,map_location='cpu',weights_only=False);assert element['geometry_sha256']==report['geometry_sha256'];assert element['field_sha256']==report['fields'][variant];bindings[str(ep)]=sha256_file(ep)
 for p in [report_path,field,parent/'frozen_geometry/report.json',Path(report['query_root'])/task/'receipt.json']:
  bindings[str(p)]=sha256_file(p)
result=dict(schema='nvos_reference_completion_next19_eight.v1',contract='nvos-ludvig-online-all-view-eight-v1',complete=True,tasks=tasks,macro=execution['macro'],per_task=rows,bindings=bindings,registration_sha256=sha256_file(root/'registration.json'),source_code_sha256=execution['script_sha256'],same_frozen_geometry_and_field=True,baseline_binary_masks_exact_all_tasks=True,negative_scribble_used=False,target_rgb_used=False,target_gt_only_after_all_variants_persisted=True,full_multiquery_method=False,independent_test=False,promotion='No automatic promotion; registered connectivity hypothesis diagnostic; no per-scene winner selection',known_limitations=['Query-time connected-instance hypothesis can remove legitimate disconnected target parts; never equivalent to verified negatives','Final positive repair can reintroduce disconnected elements; no hard connectivity guarantee','Prompt-connected reference SAM remains a teacher region, not verified target identity; can merge instances or parts','Reference SAM positives are teacher evidence, not ground truth','SAM outside is unknown and does not provide exclusion evidence','Finite optimizer budget does not mathematically guarantee prompt constraints','Only frozen supported domain; missing geometry cannot be repaired'])
out=Path('paper/artifacts/nvos_reference_completion_next19_eight.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['macro'],indent=2))
for task,row in rows.items():print(task,[(v,round(row['metrics'][v]['iou']*100,3),round(row['metrics'][v]['reference_signed_agreement']['positive_agreement']*100,3)) for v in variants])
