"""Same-field fixed calibration diagnostic, never per-scene winner selection."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS,NYU40_ID_TO_NAME
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels,volume_weighted_split_metrics
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',required=True);a=p.parse_args();torch.set_num_threads(4)
with torch.no_grad():
 for scene in a.scenes:
  root=Path('output/scannet_frozen_field_20260918')/scene;out=Path('output/scannet_calibration_diagnostic_20260919')/scene;out.mkdir(parents=True,exist_ok=False);fp=root/'reconstruction/field/bounded_fallback_admission/field.pt';report=json.loads(fp.with_name('report.json').read_text());assert sha256_file(fp)==report['field_sha256'];payload=torch.load(fp,weights_only=False,map_location='cpu');model=ElementRetrievalField(**payload['architecture']);model.load_state_dict(payload['state_dict']);bank=torch.load(root/'official_protocol_evaluation/queries.pt',weights_only=False,map_location='cpu');ids=OPENGAUSSIAN_NYU40_CLASS_SPLITS['19'];q=bank['encoding'][[bank['texts'].index(NYU40_ID_TO_NAME[c]) for c in ids]].float();cos=model.native_cosine(q);threshold=model.native_query_thresholds['text'](q);logits=model.readout_log_scale['text'].clamp(-4,6).exp()*(cos-threshold);observed=model.observed
  predictions={name:torch.where(observed,torch.tensor(ids)[score.argmax(1)],0).numpy() for name,score in [('native',cos),('calibrated_logits',logits),('calibrated_probability',logits.sigmoid())]};assert np.array_equal(predictions['native'],np.load(root/'official_protocol_evaluation/predictions.npz')['19']);np.savez_compressed(out/'predictions.npz',**predictions)
  gr=json.loads((root/'geometry_report.json').read_text())['results']['independent_scale'];geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device='cpu');xyz,labels=_read_label_ply(Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply'));gt,_=assign_vala_pseudo_labels(geo.xyz.numpy(),geo.scales.numpy(),geo.quaternions.numpy(),xyz,labels,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512);weight=(geo.scales.prod(1)*geo.opacities.flatten()).numpy();metrics={k:volume_weighted_split_metrics(gt,v,weight,ids) for k,v in predictions.items()}
  result=dict(scene=scene,diagnostic_only=True,parameters_fitted=False,selection_before_gt=True,threshold_search=False,geometry_unchanged=True,field_sha256=report['field_sha256'],native_replay_exact=True,probability_saturation_argmax_differences=int((predictions['calibrated_logits']!=predictions['calibrated_probability']).sum()),class_thresholds={str(c):float(threshold[i]) for i,c in enumerate(ids)},metrics=metrics,implementation_sha256=sha256_file(__file__));(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(scene,{k:v['miou'] for k,v in metrics.items()},flush=True)
