import pytest
import torch
from radio_gs.geometry.registered_pinhole import blender_camera_to_opencv


def test_opengl_axes_projection_and_frame_intrinsic_override():
    default=dict(w=100,h=80,fl_x=100,fl_y=100,cx=50,cy=40)
    c=blender_camera_to_opencv(dict(transform_matrix=torch.eye(4).tolist(),fl_x=200),default)
    point=torch.tensor([1.,-1.,-4.,1.],dtype=torch.float64)
    camera=c['world_to_camera']@point
    pixel=c['intrinsic']@camera[:3];pixel=pixel[:2]/pixel[2]
    assert torch.allclose(pixel,torch.tensor([100.,65.],dtype=torch.float64))
    assert torch.allclose(c['camera_to_world']@camera,point)


def test_invalid_rotation_and_missing_intrinsics_rejected():
    bad=torch.eye(4);bad[0,0]=-1
    with pytest.raises(ValueError):blender_camera_to_opencv(dict(transform_matrix=bad),{})
    with pytest.raises(ValueError):blender_camera_to_opencv(dict(transform_matrix=torch.eye(4)),{})


def test_registered_table_inverse_and_moge_receipt(tmp_path):
    import json
    from radio_gs.geometry.registered_pinhole import load_registered_camera_table
    from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
    from radio_gs.utils.immutable_artifacts import sha256_file
    cv=torch.eye(4);cv[0,3]=2
    w2c=torch.linalg.inv(cv)
    row=dict(frame_index=7,width=100,height=80,intrinsic=[[100,0,50],[0,100,40],[0,0,1]],camera_to_world=cv.tolist(),world_to_camera=w2c.tolist())
    table=tmp_path/'cameras.json';data=dict(schema='registered_pinhole_source_authority.v1',camera_convention='OpenCV: +z forward',records=[row],source_frame_indices=[7]);table.write_text(json.dumps(data))
    camera=load_registered_camera_table(table)[7]
    assert camera.world_to_camera[0,3]==-2 and not hasattr(camera,'point_ids')
    authority=tmp_path/'authority.json';authority.write_text(json.dumps(dict(frame_indices=[7])))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(records=[dict(frame_index=7)])))
    report=tmp_path/'report.json';payload=dict(geometry_receipt=dict(inputs=[dict(role=role,path=str(path),sha256=sha256_file(path)) for role,path in [('registered_camera_table',table),('source_authority',authority),('moge3_scene_manifest',manifest)]]),calibration=[dict(frame_index=7,accepted=True,scale=1,offset=0)],global_world_units_per_metre=1)
    report.write_text(json.dumps(payload));prior=MogeGeometryPrior(str(report),sha256_file(report));assert prior.scale==1 and prior.cameras[7].world_to_camera[0,3]==-2
    row['world_to_camera'][0][3]=0;table.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='inverse'):load_registered_camera_table(table)
    with pytest.raises(ValueError,match='digest'):MogeGeometryPrior(str(report),sha256_file(report))
