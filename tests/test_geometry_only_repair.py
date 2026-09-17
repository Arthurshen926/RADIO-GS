import pytest
import torch
from radio_gs.geometry.joint_scale import fit_joint_depth_scales
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.rendering.camera_clearance import quaternion_to_rotation_matrix


def test_joint_scale_recovers_consistent_multiview_depths():
    torch.manual_seed(7)
    world=torch.randn(1000,3)*.3+torch.tensor([0.,0.,3.])
    translations=torch.tensor([[0.,0.,0.],[.4,0.,0.],[-.3,.2,0.]])
    true=torch.tensor([1.,1.04,.96])
    records=[]
    for i,j in [(0,1),(0,2),(1,2)]:
        records.append((torch.full((1000,),i),torch.full((1000,),j),
                        (world-translations[i])/true[i],(world-translations[j])/true[j],
                        (translations[i]-translations[j]).expand_as(world)))
    args=[torch.cat([r[k] for r in records]) for k in range(5)]
    scale=fit_joint_depth_scales(*args,num_views=3,prior_weight=.0001)
    assert torch.allclose(scale,true,atol=.003)
    assert torch.allclose(fit_joint_depth_scales(*[x if k<2 else x*100 for k,x in enumerate(args)],num_views=3,prior_weight=.0001),scale,atol=1e-5)


def test_disconnected_view_preserves_metric_anchor():
    result=fit_joint_depth_scales(torch.tensor([0]),torch.tensor([1]),torch.tensor([[0.,0.,1.]]),
        torch.tensor([[0.,0.,1.]]),torch.zeros(1,3),num_views=3)
    assert torch.equal(result,torch.ones(3))


def test_gaussian_normal_thickness_and_frozen_state():
    normal=torch.tensor([[0.,0.,1.],[0.,0.,-1.],[1.,0.,0.]])
    model=FrozenGaussianGeometry(torch.zeros(3,3),normal,tangent_sigma=.01,normal_sigma=.001)
    assert list(model.parameters())==[]
    rotation=quaternion_to_rotation_matrix(model.get_rotation())
    assert torch.allclose(rotation[:,:,2],normal,atol=1e-6)
    assert torch.all(model.get_scaling()[:,2]<model.get_scaling()[:,0])
    assert all(not x.requires_grad for x in model.buffers())


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        FrozenGaussianGeometry(torch.zeros(1,3),torch.zeros(1,3),tangent_sigma=.01,normal_sigma=.001)


def test_visibility_change_invalidates_feature_binding():
    model=FrozenGaussianGeometry(torch.zeros(1,3),torch.tensor([[0.,0.,1.]]),tangent_sigma=.01,normal_sigma=.001)
    binding=model.fingerprint()
    model.assert_binding(binding)
    model.opacities.mul_(.5)
    with pytest.raises(ValueError,match='geometry changed'):
        model.assert_binding(binding)


def test_cross_view_support_does_not_require_same_voxel_collision():
    from radio_gs.v4.carrier import Camera
    from radio_gs.v4.geometry.tsdf_fusion import DepthObservation,SparseSurfaceFusion
    k=torch.tensor([[10.,0.,1.5],[0.,10.,1.5],[0.,0.,1.]])
    obs=[]
    for i,offset in enumerate([0.,.025]):
        pose=torch.eye(4);pose[0,3]=offset
        camera=Camera(str(i),k,pose,3,3)
        obs.append(DepthObservation(camera,torch.ones(3,3),torch.ones(3,3,dtype=torch.bool),torch.ones(3,3)))
    with pytest.raises(RuntimeError,match='rejected every'):
        SparseSurfaceFusion(.01,minimum_views=2).fuse(obs)
    fused=SparseSurfaceFusion(.01,minimum_views=2,minimum_voxel_views=1).fuse(obs)
    assert bool((fused.view_count==1).all())
    assert bool((fused.agreement_count>=2).all())
    assert bool((fused.free_space_conflict_count==0).all())


def test_colmap_pixel_centres_sample_depth_without_second_half_pixel():
    from radio_gs.v4.geometry.fuse_lerf_moge3 import _bilinear
    depth=torch.arange(12,dtype=torch.float32).reshape(3,4)
    xy=torch.tensor([[.5,.5],[1.5,1.5],[3.5,2.5]])
    assert torch.allclose(_bilinear(depth,xy,4,3),torch.tensor([0.,5.,11.]),atol=1e-6)
    assert not torch.allclose(_bilinear(depth,xy,4,3,pixel_contract='legacy_index_v1'),torch.tensor([0.,5.,11.]))


def test_rejected_view_scale_recovered_from_trusted_geometry():
    from radio_gs.geometry.joint_scale import recover_unanchored_view_scales
    h=w=32;k=torch.tensor([[20.,0.,16.],[0.,20.,16.],[0.,0.,1.]])
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    rays=torch.stack([x+.5,y+.5,torch.ones_like(x)],dim=-1).float()@torch.linalg.inv(k).T
    views=[]
    for i in range(4):
        pose=torch.eye(4);pose[0,3]=i*.01
        views.append(dict(frame=i,local_calibration_accepted=i<3,depth=torch.ones(h,w)*(2 if i<3 else 1),
                          valid=torch.ones(h,w,dtype=torch.bool),rays=rays,k=k,pose=pose,
                          normals=torch.tensor([0.,0.,1.]).expand(h,w,3)))
    result,report=recover_unanchored_view_scales(views)
    assert len(result)==4 and report[0]['accepted']
    assert report[0]['scale']==pytest.approx(2.)
    assert torch.equal(result[-1]['depth'],torch.ones(h,w)*2)
    assert torch.equal(views[-1]['depth'],torch.ones(h,w))


def test_frozen_geometry_cold_load_bit_exact(tmp_path):
    from radio_gs.utils.immutable_artifacts import sha256_file
    model=FrozenGaussianGeometry(torch.zeros(1,3),torch.tensor([[1.,0.,0.]]),tangent_sigma=.01,normal_sigma=.001)
    path=tmp_path/'geometry.pt'
    torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',state_dict=model.state_dict(),
                    geometry_trainable_during_semantics=False),path)
    restored=FrozenGaussianGeometry.load(path,expected_sha256=sha256_file(path))
    restored.assert_binding(model.fingerprint())
    assert not list(restored.parameters())


def test_nonfinite_confidence_cannot_poison_geometry():
    from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
    model=FrozenGaussianGeometry(torch.tensor([[0.,0.,1.]]),torch.tensor([[0.,0.,1.]]),tangent_sigma=.01,normal_sigma=.001)
    renderer=FeatureFieldRenderer(2,2,2.,2.,1.,1.)
    with pytest.raises(ValueError,match='finite'):
        renderer.render_feature_rows(model,torch.eye(4),torch.ones(1,1),row_confidence=torch.tensor([float('nan')]))


def test_corrected_normals_remain_perpendicular_to_corrected_surface():
    from radio_gs.geometry.ray_alignment import align_normals_to_intrinsics
    source=torch.diag(torch.tensor([2.,3.,1.]));target=torch.eye(3)
    n=torch.tensor([[1.,1.,1.]])
    tangent=torch.tensor([[1.,-1.,0.]])
    corrected=align_normals_to_intrinsics(n,source,target)
    corrected_tangent=tangent@(torch.linalg.inv(target)@source).T
    assert torch.allclose((corrected*corrected_tangent).sum(1),torch.zeros(1),atol=1e-6)


def test_geometry_cache_precision_and_resolution_are_resume_contract(tmp_path):
    from radio_gs.v4.geometry.batch_moge3 import _existing_output_is_valid
    path=tmp_path/'prediction.pt'
    torch.save({'geometry_dtype':'float16','resolution_level':5},path)
    assert not _existing_output_is_valid(path,{'geometry_dtype':'float32','resolution_level':5})
    assert not _existing_output_is_valid(path,{'geometry_dtype':'float16','resolution_level':9})


def test_camera_cache_identity_includes_pose_and_raster(tmp_path):
    from radio_gs.v4.carrier import Camera,GaussianCarrier
    path=tmp_path/'view.pt'
    torch.save(dict(gaussian_ids=torch.tensor([0]),pixel_ids=torch.tensor([0]),base_weights=torch.tensor([1.]),num_gaussians=1,num_pixels=4),path)
    c1=Camera('same',torch.eye(3),torch.eye(4),2,2)
    c2=Camera('same',torch.eye(3),torch.eye(4),1,4)
    assert c1.projection_key!=c2.projection_key
    carrier=GaussianCarrier(1,{'same':path});carrier.project(c1)
    with pytest.raises(ValueError,match='camera changed'):carrier.project(c2)


def test_pose_scale_must_not_be_hidden_in_rotation():
    from radio_gs.v4.carrier import Camera
    pose=torch.eye(4);pose[:3,:3]*=2
    with pytest.raises(ValueError,match='rigid'):Camera('scaled',torch.eye(3),pose,2,2)


def test_feature_confidence_preserves_occlusion_and_gradient(monkeypatch):
    from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
    model = FrozenGaussianGeometry(torch.tensor([[0., 0., 1.], [0., 0., 2.]]),
        torch.tensor([[0., 0., 1.], [0., 0., 1.]]),
        tangent_sigma=.01, normal_sigma=.001)
    renderer = FeatureFieldRenderer(1, 1, 1., 1., .5, .5)

    def composite(means, rotations, scales, opacity, features, *args):
        # Front-to-back alpha compositing: the rear contribution is occluded.
        weights = torch.stack((opacity[0], (1-opacity[0])*opacity[1]))
        rendered = (weights[:, None]*features).sum(0)[None, None, None]
        alpha = weights.sum().reshape(1, 1, 1, 1)
        depth = ((weights*means[:, 2]).sum()/weights.sum()).reshape(1, 1, 1, 1)
        return rendered, depth, alpha

    monkeypatch.setattr(renderer, "_chunk_render", composite)
    values = torch.tensor([[2.], [8.]], requires_grad=True)
    baseline = renderer.render_feature_rows(model, torch.eye(4), values, alpha_normalize=True)
    weighted = renderer.render_feature_rows(model, torch.eye(4), values,
        alpha_normalize=True, row_confidence=torch.tensor([.5, 1.]))
    assert torch.equal(baseline["alpha_map"], weighted["alpha_map"])
    assert torch.equal(baseline["depth_map"], weighted["depth_map"])
    assert torch.allclose(weighted["feature_map"], torch.tensor([[[26/7]]]))
    weighted["feature_map"].sum().backward()
    assert torch.allclose(values.grad[:, 0], torch.tensor([5/7, 2/7]))
    missing = renderer.render_feature_rows(model, torch.eye(4), values,
        alpha_normalize=True, row_confidence=torch.zeros(2))
    assert torch.count_nonzero(missing["feature_map"]) == 0
    assert torch.count_nonzero(missing["feature_support_map"]) == 0
    assert torch.equal(missing["alpha_map"], baseline["alpha_map"])


def test_affine_bias_does_not_fill_missing_geometry_or_evidence():
    from types import SimpleNamespace
    from radio_gs.rendering.coefficient_renderer import decode_visible_coefficients
    field = SimpleNamespace(decoder=SimpleNamespace(decode_map=lambda x: x + 3))
    geometry = SimpleNamespace(feature_confidence_contract="post_visibility_normalized_v1")
    compact = {"feature_map": torch.ones(1, 1, 3), "alpha_map": torch.tensor([[1., 0., 1.]]),
               "feature_support_map": torch.tensor([[1., 1., 0.]])}
    assert torch.equal(decode_visible_coefficients(field, compact, geometry),
                       torch.tensor([[[4., 0., 0.]]]))
    # Historical replay keeps the original affine background contract.
    assert torch.equal(decode_visible_coefficients(field, compact, object()),
                       torch.full((1, 1, 3), 4.))


def test_exact_compositor_cannot_bypass_frozen_opacity_contract():
    from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
    from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
    model = FrozenGaussianGeometry(torch.tensor([[0.,0.,1.]]), torch.tensor([[0.,0.,1.]]),
                                   tangent_sigma=.01, normal_sigma=.001)
    renderer = FeatureFieldRenderer(1,1,1.,1.,.5,.5)
    with pytest.raises(ValueError, match="forbids opacity scaling"):
        rasterize_single_view_contributions(model, renderer, torch.eye(4),
                                            height=1, width=1, opacity_scale=torch.ones(1))
