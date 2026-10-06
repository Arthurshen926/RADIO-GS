"""Exact source-only materialization of the frozen raw-native classification path.

No structural/image/prompt head completion is claimed. Optional reference checks
prove equivalence on an already trained scene before using new source cohorts.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
def load(p,h=None):
 if h:assert sha256_file(p)==h
 return torch.load(p,map_location='cpu',weights_only=False)
r=a.root;info=json.loads((r/'teachers.json').read_text());ep=load(r/'episodes.pt',json.loads((r/'episodes.json').read_text())['sha256']);assert ep['teacher_manifest_sha256']==sha256_file(r/'teachers.json');init=load(r/'source_fusion_initialization.pt',json.loads((r/'source_fusion_initialization.json').read_text())['sha256']);assert init['teacher_manifest_sha256']==sha256_file(r/'teachers.json') and init['geometry_binding']==ep['geometry_binding'];basis=init['basis'].T;valid=init['valid'];n=len(valid);rec=info['teacher_files']['siglip'];desc=F.normalize(load(r/rec['path'],rec['sha256'])['descriptors'].float(),dim=1)@basis;auth=json.loads((r/'responsibility.json').read_text());assert sha256_file(r/'responsibility.json')==info['authority_sha256'];num=torch.zeros(n,64);mass=torch.zeros(n)
for frame in info['train_frames']:
 vi=info['frames'].index(frame);ids=torch.where(ep['proposal_views']==vi)[0];rec=auth['views'][vi];hit=load(r/rec['relative_path'],rec['sha256']);w,_=source_pixel_evidence_weights(ep['masks'][ids].flatten(1),hit['pixel_ids'].long(),hit['gaussian_ids'].long(),hit['base_weights'].float(),n,partition_overlap=True);num+=w.T@desc[ids];mass+=w.sum(0)
assert (mass[valid]>0).all();latent=num/mass[:,None].clamp_min(1e-8);bank=load(a.queries)
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS,NYU40_ID_TO_NAME
preds={};q=F.normalize(bank['encoding'].float(),dim=1);scores=F.normalize(latent@basis.T,dim=1)@q.T
for name,ids in OPENGAUSSIAN_NYU40_CLASS_SPLITS.items():
 if name not in ['19','15','10']:continue
 cols=[bank['texts'].index(NYU40_ID_TO_NAME[c]) for c in ids];preds[name]=torch.where(valid,torch.tensor(ids)[scores[:,cols].argmax(1)],0).numpy()
np.savez_compressed(a.output/'predictions.npz',**preds);torch.save(dict(native_latent=latent,native_decoder=basis,observed=valid,geometry_binding=init['geometry_binding']),a.output/'native_field.pt');checks=None
if a.reference:
 original=np.load(a.reference);checks={k:int((v!=original[k]).sum()) for k,v in preds.items()};assert not any(checks.values()),checks
report=dict(source_only_construction=True,benchmark_ground_truth_opened=False,other_query_heads_built=False,native_classification_only=True,geometry_binding=init['geometry_binding'],observed_elements=int(valid.sum()),elements=n,reference_prediction_mismatches=checks,teacher_manifest_sha256=sha256_file(r/'teachers.json'),initialization_sha256=sha256_file(r/'source_fusion_initialization.pt'),query_sha256=sha256_file(a.queries),prediction_sha256=sha256_file(a.output/'predictions.npz'),field_sha256=sha256_file(a.output/'native_field.pt'),implementation_sha256=sha256_file(__file__));(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.output,checks,flush=True)
