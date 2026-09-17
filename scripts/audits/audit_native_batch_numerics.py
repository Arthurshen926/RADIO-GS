"""Separate FP16 encoder batch numerics from scene-field query batching."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
f=torch.load(a.root/'cold_query_fixture.pt',weights_only=False);assert sha256_file(f['image']['path'])==f['image']['sha256'];enc=NativeThreeQueryEncoder(f['encoder_identity'],a.device)
q1=enc.encode_image(Image.open(f['image']['path'])).half().float();rgb=pil_to_tensor(Image.open(f['image']['path']).convert('RGB').resize((384,384),Image.Resampling.BICUBIC)).float()[None]/255
q8=enc.vision.encode(rgb.repeat(8,1,1,1)).half().float();expected=f['expected_encodings']['image'].float().to(a.device)
geo=FrozenGaussianGeometry.load(f['geometry_path'],expected_sha256=f['geometry_sha256'],device=a.device);r=json.loads((a.root/'nvos96_protected_imagecal_balanced/report.json').read_text());field,_=load_element_retrieval_field(a.root/'nvos96_protected_imagecal_balanced/field.pt',expected_sha256=r['field_sha256'],geometry=geo,expected_query_encoder=f['encoder_identity'],device=a.device)
with torch.no_grad():
 base=field.retrieve(q1,'image');batch=field.retrieve(q8,'image');reference=field.retrieve(expected,'image');samequerybatch=field.retrieve(q1.repeat(8,1),'image');valid=base['valid'];s=base['scores'][:,0]
 def comparison(q,pred):
  other=pred['scores'][:,0];return dict(encoding_cosine=float(torch.nn.functional.cosine_similarity(q,expected).mean()),max_posterior_difference=float((s-other).abs().max()),mean_posterior_difference=float((s[valid]-other[valid]).abs().mean()),supported_threshold_flip_fraction=float(((s[valid]>=.5)!=(other[valid]>=.5)).float().mean()))
 result=dict(schema='native_batch_numerics_audit.v1',field_sha256=r['field_sha256'],fixture_sha256=sha256_file(a.root/'cold_query_fixture.pt'),image_query_only=True,benchmark_ground_truth_used=False,single_encoding_cosine_to_fixture=float(torch.nn.functional.cosine_similarity(q1,expected)),batch8_encoding_vs_single=comparison(q8[:1],batch),fixture_encoding_vs_single=comparison(expected,reference),identical_encoded_query_field_batch_vs_single=comparison(q1,samequerybatch),implementation_sha256=sha256_file(__file__),limitations='One source fixture crop, FP16 GPU inference; no final query quality measured and no encoder identity altered')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
