"""Fresh native image encoding and scene retrieval with training caches blocked."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.scripts.build_native_dinov2_exact_mpr_teacher import _load_model,_extract_feature_map
from radio_gs.utils.immutable_artifacts import sha256_file
def reject_training(event,args):
 if event=='open' and args and isinstance(args[0],str):
  path=args[0].lower()
  if any(x in path for x in ('canonical_teacher_features','c-radio','c_radio_v','/features/frame_','/dino.pt')):
   raise RuntimeError('training/RADIO artifact access forbidden: '+path)
sys.addaudithook(reject_training)
torch.set_num_threads(4);device=torch.device('cuda:0')
root=Path('/mnt/pool/sqy/results/RADIO-GS/output/frozen_native_dino_pilot_v2_20260915/figurines')
report=json.loads(Path('paper/artifacts/native_image_retrieval_pilot_support_v2_20260915.json').read_text())
geometry_path=Path('/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914/figurines/independent_scale.pt')
geometry=FrozenGaussianGeometry.load(geometry_path,expected_sha256='870232321b96a24e1fb943c7554a10f78cb1453a6c4ab17d7c2319cd9196554b',device=device)
checkpoint=Path('/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth')
identity={'model':'dinov2_vitb14','sha256':sha256_file(checkpoint),'normalization':'l2_native_patch'}
field,payload=load_element_retrieval_field(report['checkpoint'],expected_sha256=report['checkpoint_sha256'],
 geometry=geometry,expected_query_encoder=identity,device=device)
encoder=_load_model(SimpleNamespace(checkpoint=str(checkpoint),torchhub_repo='/root/.cache/torch/hub/facebookresearch_dinov2_main',model_name='dinov2_vitb14'),device)
rgb=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/figurines/images/frame_00018.jpg')
feature=_extract_feature_map(encoder,rgb,device=device,grid_height=46,grid_width=62,patch_size=14)
query=feature[:,23,31][None]
with torch.no_grad():result=field.retrieve(query,'image')
assert result['scores'].shape==(len(geometry.xyz),1) and torch.isfinite(result['scores']).all()
geometry.assert_binding(payload['geometry_binding'])
out={'role':'cold_native_image_query_inference','radio_and_training_feature_access_blocked':True,
 'encoder_loaded_from_native_checkpoint':True,'source_query_rgb_sha256':sha256_file(rgb),
 'query_patch_yx':[23,31],'geometry_binding_verified':True,'field_sha256':report['checkpoint_sha256'],
 'num_elements':len(geometry.xyz),'elements_with_training_evidence':int(result['valid'].sum()),
 'score_min':float(result['scores'].min()),'score_max':float(result['scores'].max()),
 'semantic_accuracy_measured':False}
Path('paper/artifacts/cold_native_retrieval_20260915.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
