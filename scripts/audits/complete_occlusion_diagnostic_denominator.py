"""Restore empty selections to diagnostic means; official evaluation is unchanged."""
import json
from pathlib import Path
import numpy as np
import torch
from radio_gs.scripts.eval_opengaussian_lerf_baseline import load_binary_mask
from radio_gs.utils.immutable_artifacts import sha256_file
for scene in ['figurines','ramen']:
 root=Path('output/lerf_selected_ladder_20260919')/scene;src=root/'occlusion_attribution.json';out=root/'occlusion_attribution_complete.json';assert not out.exists();d=json.loads(src.read_text());ref=json.loads((root/'report.json').read_text());sel=torch.load(root/'selections.pt',weights_only=False)['final'];data=torch.load(Path('output/lerf_semantic_ladder_calibrated_20260919')/scene/'scores.pt',weights_only=False);seen={(r['frame'],r['query']) for r in d['observations']};added=0
 for obj in ref['results']['final']['objects']:
  if (obj['frame'],obj['query']) in seen:continue
  assert not sel[:,data['texts'].index(obj['query'])].any() and obj['iou']==0
  d['observations'].append(dict(frame=obj['frame'],query=obj['query'],selected_elements=0,fp_exposed_by_removing_nonselected=0,full_scene_selected_contribution_iou=0.,selected_only_iou=0.,centers_project_inside=0,false_positive_pixels=0,false_positive_alpha_mass=[0.,0.],fp_pixels_with_inside_center_contribution_above_threshold=0,predicted_pixels=0,target_pixels=int(load_binary_mask(Path(obj['gt_path'])).sum())));added+=1
 d['selected_only_miou']=float(np.mean([r['selected_only_iou'] for r in d['observations']]));d['full_scene_selected_contribution_miou']=float(np.mean([r['full_scene_selected_contribution_iou'] for r in d['observations']]));assert len(d['observations'])==ref['results']['final']['count'];assert abs(d['selected_only_miou']-ref['results']['final']['miou'])<1e-7
 d.update(empty_selections_restored=added,all_official_observations_in_denominator=True,parent_report_sha256=sha256_file(src),denominator_correction_implementation_sha256=sha256_file(__file__));out.write_text(json.dumps(d,indent=2)+'\n');print(scene,added,d['selected_only_miou'],d['full_scene_selected_contribution_miou'])
