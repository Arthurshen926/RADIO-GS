"""A constructive counterexample to commuting historical hybrid decoding/raster.

Random initialized module values are an operator check, never a quality score.
"""
import json
from pathlib import Path
import torch
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
from radio_gs.utils.immutable_artifacts import sha256_file

torch.manual_seed(42)
model=HybridFeatureGaussian(latent_dim=4,hash_output_dim=4,fine_dim=4,coarse_dim=4,output_dim=8,
 num_levels=2,log2_hashmap_size=5,base_resolution=2,max_resolution=4,fine_hidden_dim=8,coarse_hidden_dim=8,fusion_hidden_dim=8)
z=torch.tensor([[1.,-2.,3.,-4.],[-4.,3.,-2.,1.]])
p=torch.tensor([[.1,.1,.1],[.9,.9,.9]])
with torch.no_grad():
 point=model._decode_point_features(z,p)
 mixed=model._decode_point_features(z.mean(0,keepdim=True),p.mean(0,keepdim=True))
 error=float((mixed-point.mean(0)).abs().max())
 assert error>1e-6
result=dict(schema='historical_hybrid_operator_audit.v1',synthetic=True,not_a_performance_metric=True,
 training_entry='radio_gs/scripts/train_feature_field.py: hybrid decode after rasterized latent and unprojected depth',
 point_entry='radio_gs/models/hybrid_gaussian.py: _decode_point_features -> decode_screen_space on individual points',
 maximum_difference=error,seed=42,
 conclusion='Nonlinear decode and hashing after raster do not commute with per-element decode then raster. Baking requires its own training contract, not copying screen-space supervision unchanged.',
 source_identity={p:sha256_file(p) for p in [__file__,'radio_gs/models/hybrid_gaussian.py','radio_gs/scripts/train_feature_field.py']})
p=Path('paper/artifacts/historical_hybrid_operator_audit_20260917.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(result)
