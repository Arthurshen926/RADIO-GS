from types import SimpleNamespace
import pytest
import torch
from radio_gs.geometry.source_geometry_override import load_source_geometry_override
from radio_gs.scripts.repair_moge_geometry import load_views
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file


def setup_prior(tmp_path):
 camera=SimpleNamespace(intrinsic=torch.eye(3),width=2,height=2,world_to_camera=torch.eye(4),camera_to_world=torch.eye(4))
 raw=tmp_path/'raw.pt';torch.save({'unused_raw_depth':100.},raw);rawsha=sha256_file(raw)
 path=tmp_path/'map.pt';torch.save(dict(schema='source_local_scale_geometry.v1',frame=1,prediction_sha256=rawsha,source_only=True,benchmark_data_used=False,depth=torch.full((2,2),2.),normals=torch.tensor([0.,0.,1.]).expand(2,2,3),valid=torch.ones(2,2,dtype=torch.bool),intrinsic=torch.eye(3),world_to_camera=torch.eye(4)),path)
 record=dict(path=str(path),sha256=sha256_file(path))
 p=object.__new__(MogeGeometryPrior);p.records={1:dict(prediction=str(raw),prediction_sha256=rawsha)};p.calibration={1:dict(accepted=True,scale=99.,offset=100.,source_geometry_override=record)};p.cameras={1:camera};p.scale=1.;p.strength=.5;p.statistics=[]
 return p,record,camera,rawsha


def test_geometry_loader_and_depth_conditioner_use_identical_materialized_depth(tmp_path):
 p,record,camera,digest=setup_prior(tmp_path)
 v=load_views(p,2,'cpu')[0]
 assert torch.equal(v['depth'],torch.full((2,2),2.)) # No second scale/offset application.
 out=p.apply(frame_index=1,xyz=torch.tensor([[.5,.5,2.]]),world_to_camera=torch.eye(4),intrinsic=torch.eye(3),gaussian_ids=torch.tensor([0]),pixel_ids=torch.tensor([0]),weights=torch.ones(1),height=2,width=2)
 assert torch.equal(out,torch.ones(1))
 with pytest.raises(ValueError,match='width differs'):load_views(p,3,'cpu')


def test_override_rejects_wrong_source_or_camera(tmp_path):
 p,record,camera,digest=setup_prior(tmp_path)
 with pytest.raises(ValueError,match='provenance'):load_source_geometry_override(record,frame=1,prediction_sha256='wrong',camera=camera)
 camera.intrinsic[0,0]=2
 with pytest.raises(ValueError,match='camera'):load_source_geometry_override(record,frame=1,prediction_sha256=digest,camera=camera)
