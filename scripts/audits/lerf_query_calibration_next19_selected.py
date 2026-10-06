"""Frozen same-carrier selection ladder; no query-specific fitting or GT selection."""
import argparse,json
from pathlib import Path
from dataclasses import asdict
import torch
from PIL import Image
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES,evaluate_scene
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 src=Path('output/lerf_query_calibration_next19')/a.scene;prep=json.loads((src/'preparation.json').read_text());assert sha256_file(src/'scores.pt')==prep['scores_sha256'];data=torch.load(src/'scores.pt',weights_only=False)
 gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';assert sha256_file(gp)==prep['geometry_report_sha256'];gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(data['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
 a.output.mkdir(parents=True,exist_ok=False)
 selections={k:(v>.5)&data['observed'][:,None] for k,v in data['scores'].items() if k=='final' or k.endswith('_calibrated')};torch.save(selections,a.output/'selections.pt')
 frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene,frames=SCENE_GT_FRAMES[a.scene]);result={}
 for name,selected in selections.items():
  dst=a.output/name;dst.mkdir();selected=selected.cuda()
  for frame,objects in frames.items():
   camera=prior.cameras[int(frame.split('_')[-1])];h,w=objects[0].mask.shape;k=camera.intrinsic.cuda().clone();k[0]*=w/camera.width;k[1]*=h/camera.height
   for obj in objects:
    ids=selected[:,data['texts'].index(obj.query)]
    if ids.any():
     _,alpha,_=rasterization(means=geo.xyz[ids],quats=geo.quaternions[ids],scales=geo.scales[ids],opacities=geo.opacities.flatten()[ids],colors=torch.ones(int(ids.sum()),1,device='cuda:0'),viewmats=camera.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
     byte=(alpha[0,...,0]*255+.5).floor().clamp(0,255).byte().cpu().numpy()
    else:byte=torch.zeros(h,w,dtype=torch.uint8).numpy()
    Image.fromarray(byte).save(dst/(frame+'_'+obj.query+'.png'))
  gt=Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/a.scene/'gt';result[name]=asdict(evaluate_scene(gt,dst,a.scene,threshold=10));assert result[name]['missing']==0;print(a.scene,name,result[name],flush=True)
 report=dict(scene=a.scene,diagnostic_only=True,geometry_unchanged=True,threshold_search=False,source_only_scores=True,target_rgb_used=False,selection_independent_of_target=True,volumetric_3d_iou=False,gt='frozen historical JPEG masks',scores_sha256=prep['scores_sha256'],selection_sha256=sha256_file(a.output/'selections.pt'),implementation_sha256=sha256_file(__file__),results=result)
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 with torch.no_grad():main()
