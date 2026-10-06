"""Fixed-contract full-cohort text evaluation of source-only uplift controls."""
import argparse,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from sklearn.metrics import average_precision_score
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects,_mask_iou
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES,evaluate_scene
from radio_gs.utils.immutable_artifacts import sha256_file


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
    base=Path('output/full_field_rebuild_next19')/a.scene/'evaluation';root=Path('output/query_region_readout_next25')/a.scene;out=root/'evaluation';out.mkdir(exist_ok=False)
    report=json.loads((root/'preparation.json').read_text());assert report['complete'];assert sha256_file(root/'scores.pt')==report['scores_sha256']
    old=json.loads((base/'report.json').read_text());assert old['complete'];assert sha256_file(base/'scores.pt')==old['scores_sha256'];assert sha256_file(base/'text_online.encodings.pt')==old['encoding_bank_sha256']
    data=torch.load(root/'scores.pt',map_location='cpu',weights_only=False)
    assert data['geometry_binding']==report['geometry_binding'];assert old['field_sha256']==report['field_sha256']
    scores=data['scores'];assert list(scores)==report['arms'] and torch.equal(scores['final'],torch.load(base/'scores.pt',map_location='cpu',weights_only=False)['scores']['final'])
    gp=Path('output/depth_opacity_repair_20260918')/a.scene/'geometry_report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda:0');geo.assert_binding(data['geometry_binding']);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    frames=load_lerf_objects(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')/a.scene,frames=SCENE_GT_FRAMES[a.scene]);assert {int(f.split('_')[-1]) for f in frames}.isdisjoint(report['source_views'])
    obs=data['observed'].cuda();rows={n:[] for n in scores};selected={};baseline_disagreement=0
    for name,values in scores.items():
        dst=out/name;dst.mkdir();values=values.cuda()
        for frame,objects in frames.items():
            cam=prior.cameras[int(frame.split('_')[-1])];h,w=objects[0].mask.shape;k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
            kw=dict(viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            for obj in objects:
                qi=data['texts'].index(obj.query);colors=torch.stack([values[:,qi],obs.float()],1)
                im,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,**kw)
                mass=im[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);prob=torch.where(valid,im[0,...,0]/mass.clamp_min(1e-8),0).cpu();binary=prob>.5
                if name=='final':
                    reference=np.asarray(Image.open(base/'final/lerf2d'/(frame+'_'+obj.query+'.png')))>0
                    baseline_disagreement+=int((reference!=binary.numpy()).sum())
                rows[name].append(dict(frame=frame,query=obj.query,iou=_mask_iou(torch.from_numpy(obj.mask),binary),ap=float(average_precision_score(obj.mask.flatten(),prob.numpy().flatten()))))
                ids=(values[:,qi]>.5)&obs
                if ids.any():
                    _,aa,_=rasterization(means=geo.xyz[ids],quats=geo.quaternions[ids],scales=geo.scales[ids],opacities=geo.opacities.flatten()[ids],colors=torch.ones(int(ids.sum()),1,device='cuda'),**kw);byte=(aa[0,...,0]*255+.5).floor().clamp(0,255).byte().cpu().numpy()
                else:byte=np.zeros((h,w),dtype=np.uint8)
                Image.fromarray(byte).save(dst/(frame+'_'+obj.query+'.png'))
            print(a.scene,name,frame,flush=True)
        selected[name]=asdict(evaluate_scene(Path('output/protocol_audit_20260801/vala/lerf3d_occam_geometry_v1/eval_gt')/a.scene/'gt',dst,a.scene,threshold=10));assert selected[name]['missing']==0
    assert baseline_disagreement==0
    def comparable(result):
        return dict(result,objects=[{k:v for k,v in row.items() if k!='pred_path'} for row in result['objects']])
    assert comparable(selected['final'])==comparable(old['selected']['final'])
    result=dict(complete=True,baseline_disagreement_pixels=baseline_disagreement,lerf2d={n:dict(miou=float(np.mean([r['iou'] for r in rr])),ap=float(np.mean([r['ap'] for r in rr])),observations=rr) for n,rr in rows.items()},selected=selected,preparation_sha256=sha256_file(root/'preparation.json'),scores_sha256=report['scores_sha256'],default_promoted=False,independent_test=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,{n:r['miou'] for n,r in result['lerf2d'].items()},{n:r['miou'] for n,r in selected.items()},flush=True)


if __name__=='__main__':main()
