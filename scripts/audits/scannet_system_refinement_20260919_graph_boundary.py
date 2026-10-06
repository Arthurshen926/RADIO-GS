"""Strict 16NN semantic transition diagnostic, not official boundary metric."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels
from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
from radio_gs.utils.immutable_artifacts import sha256_file
torch.set_num_threads(4);root=Path('output/scannet_system_refinement_20260919_graph');out=root/'boundary_diagnostic.json';assert not out.exists();summary=json.loads((root/'summary.json').read_text());results={}
for scene in summary['scenes']:
 base=Path('output/scannet_frozen_field_20260918')/scene;gr=json.loads((base/'geometry_report.json').read_text())['results']['independent_scale'];geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256']);xyz,lab=_read_label_ply(Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply'));gt,_=assign_vala_pseudo_labels(geo.xyz.numpy(),geo.scales.numpy(),geo.quaternions.numpy(),xyz,lab,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512);_,nn=cKDTree(geo.xyz.numpy()).query(geo.xyz.numpy(),k=17,workers=4);neighbors=np.stack([n[n!=i][:16] for i,n in enumerate(nn)]);rows={}
 for split in ['19','15','10']:
  valid=np.isin(gt,SPLITS[split]);edge=valid[:,None]&valid[neighbors];truth=(gt[:,None]!=gt[neighbors])&edge;rows[split]={}
  for name in ['native','bounded_logit','spatial_graph','structure32_graph']:
   pred=np.load(root/scene/(name+'.npz'))[split];boundary=(pred[:,None]!=pred[neighbors])&edge;tp=int((boundary&truth).sum());fp=int((boundary&~truth&edge).sum());fn=int((~boundary&truth).sum());rows[split][name]=dict(f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),tp=tp,fp=fp,fn=fn,valid_directed_edges=int(edge.sum()))
 results[scene]=rows;print(scene,flush=True)
macro={s:{name:{metric:float(np.mean([results[scene][s][name][metric] for scene in summary['scenes']])) for metric in ['f1','precision','recall']} for name in ['native','bounded_logit','spatial_graph','structure32_graph']} for s in ['19','15','10']};out.write_text(json.dumps(dict(diagnostic_only=True,not_official_metric=True,not_training_supervision=True,definition='Unweighted strict label-transition F1 on fixed16 spatial nearest-neighbor directed edges of ALL geometry elements, requiring both endpoints GT classes in split. No tolerance, no boundary matching. Labels same VALA pseudoGT; not independent physical boundary truth.',results=results,macro=macro,implementation_sha256=sha256_file(__file__)),indent=2)+'\n');print(macro)
