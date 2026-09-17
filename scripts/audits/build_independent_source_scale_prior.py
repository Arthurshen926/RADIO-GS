"""Build a source-only depth prior without assuming monocular scales nearly agree.

No fusion/global-scale fallback: rejected views remain rejected. Point IDs used
for final audit are disjoint from fit and admission; sparse evidence is not dense GT.
"""
import argparse,json
from pathlib import Path
import torch
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_images,_read_cameras_binary,_read_strict_source_points,_bilinear
from radio_gs.geometry.source_camera_assets import source_colmap_assets
from radio_gs.geometry.independent_view_scale import fit_independent_view_scale
from radio_gs.v4.contracts.geometry_receipt import GeometryReceipt,HashedInput
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
 manifest=json.loads(a.scene_manifest.read_text());authority_path=Path(manifest['source_authority']);assert sha256_file(authority_path)==manifest['source_authority_sha256'];authority=json.loads(authority_path.read_text())
 if any(manifest[k] for k in ['target_rgb_opened','benchmark_images_opened','benchmark_masks_opened','text_queries_opened']):raise ValueError('source-only predictions required')
 sparse,sidecar=source_colmap_assets(manifest['scene_root']);cams=_read_images(sparse/'images.bin',_read_cameras_binary(sparse/'cameras.bin'),authority.get('camera_name_to_frame_index'))
 frames=set(map(int,authority['frame_indices']));records={int(r['frame_index']):r for r in manifest['records']};assert set(records)==frames
 points,pfilter=_read_strict_source_points(sparse/'points3D.bin',{cams[f].image_id for f in frames},maximum_reprojection_error=2.,minimum_track_views=2)
 fits=[]
 for frame,rec in sorted(records.items()):
  assert sha256_file(rec['prediction'])==rec['prediction_sha256'];blob=torch.load(rec['prediction'],weights_only=False);assert blob['model_checkpoint_sha256']==manifest['model_checkpoint_sha256'];cam=cams[frame]
  indices=[i for i,pid in enumerate(cam.point_ids.tolist()) if pid in points]
  if not indices:
   fits.append(dict(frame_index=frame,accepted=False,scale=None,reason='no_reliable_source_tracks',fit_points=0,gate_points=0,audit_points=0));continue
  ids=cam.point_ids[indices];xy=cam.xy[indices];xyz=torch.stack([points[int(pid)] for pid in ids]);reference=(xyz@cam.world_to_camera[:3,:3].T+cam.world_to_camera[:3,3])[:,2]
  predicted=_bilinear(blob['point_map'][...,2].float(),xy,cam.width,cam.height,pixel_contract='image_edge_v2');valid=_bilinear(blob['validity'].float(),xy,cam.width,cam.height,pixel_contract='image_edge_v2')>.999
  fit=fit_independent_view_scale(predicted,reference,ids,valid);fits.append(dict(frame_index=frame,**fit));print(frame,fit['accepted'],fit.get('gate'),flush=True)
 accepted=[f['scale'] for f in fits if f['accepted']];fitted=[f['scale'] for f in fits if f['scale'] is not None]
 unit=float(torch.tensor(accepted or fitted).median()) if fitted else 1.
 calibration=[dict(frame_index=f['frame_index'],accepted=f['accepted'],scale=f['scale']/unit if f['scale'] is not None else 1.,offset=0.,sample_count=f['fit_points'],median_absolute_residual=None,rejection_reason=f['reason'],source_scale_fit=f) for f in fits]
 policy=dict(contract='source_point_id_scale_v1',fit='point ID modulo5 !=0',admission='point ID modulo10 ==0; >=16; median<=.05; p90<=.15',audit='point ID modulo10 ==5; never used for fitting/admission',track_policy='complete track inside legal source frames, reprojection error<=2px, >=2 views',maximum_local_scale_deviation=None,global_scale_fallback_view_count=0,global_scale_is_estimated_moge_metric_gauge_not_metric_ground_truth=True)
 inputs=[HashedInput.seal('moge3_scene_manifest',a.scene_manifest),HashedInput.seal('source_authority',authority_path),HashedInput.seal('colmap_cameras',sparse/'cameras.bin'),HashedInput.seal('colmap_images',sparse/'images.bin'),HashedInput.seal('colmap_sparse_points',sparse/'points3D.bin')]
 if sidecar is not None:inputs.append(HashedInput.seal('released_image_intrinsics',sidecar))
 receipt=GeometryReceipt(carrier='calibrated_source_depth_prior_no_fusion',coordinate_convention='colmap_world_opencv_camera_pixel_centres',inputs=tuple(inputs),source_rgb_opened=False,target_rgb_opened=False,benchmark_images_opened=False,benchmark_masks_opened=False,benchmark_labels_opened=False,model_family='MoGe-3',model_checkpoint_sha256=manifest['model_checkpoint_sha256'],calibration=policy,metadata={'scene_label':manifest['scene_label'],'surface_built':False})
 from dataclasses import asdict
 result=dict(schema='radio_gs.independent_source_scale_prior.v1',scene_label=manifest['scene_label'],global_colmap_units_per_metre=unit,calibration=calibration,calibration_pixel_contract='image_edge_v2',geometry_receipt=asdict(receipt),accepted_view_count=len(accepted),rejected_view_count=len(fits)-len(accepted),global_scale_fallback_view_count=0,point_filter=pfilter,source_only=True,benchmark_data_used=False,implementation_sha256=sha256_file(__file__),fit_implementation_sha256=sha256_file('radio_gs/geometry/independent_view_scale.py'))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print('accepted',len(accepted),'of',len(fits),flush=True)
if __name__=='__main__':main()
