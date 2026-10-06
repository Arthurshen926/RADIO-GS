"""Native image ranking control under the frozen positive-only NVOS query rights."""
import argparse
import json
from pathlib import Path
import sys
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scenes',nargs='+',required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    mp=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json')
    manifest=json.loads(mp.read_text())
    forbidden=set()
    targets=set()
    for item in manifest['scenes']:
        forbidden.add(str(Path(item['prompt']['negative_path']).resolve()))
        for frame in item['frames']:
            if frame['frame_id'] in item['evaluation_frame_ids']:
                forbidden.update(str(Path(frame[k]).resolve()) for k in ('rgb_path','annotation_rgb_path'))
                targets.add(str(Path(frame['gt_mask_path']).resolve()))
    saved=False
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=str(Path(os.fsdecode(args[0])).resolve())
            if path in forbidden or (path in targets and not saved):raise RuntimeError('Unauthorized target or negative evidence')
            if any(x in path for x in ('teacher_siglip.pt','teacher_sam.pt','teacher_dino.pt','teacher_radio.pt','episodes.pt')):
                raise RuntimeError('Training assets forbidden at query time')
    sys.addaudithook(guard)
    encoder=None
    for name in a.scenes:
        saved=False
        scene=next(s for s in manifest['scenes'] if s['scene_id']==name)
        target=next(f for f in scene['frames'] if f['frame_id'] in scene['evaluation_frame_ids'])
        mapname='horns_center' if name=='horns_left' else name
        folder=Path('output/native_residual_basis_next30_v2')/('nvos_'+mapname)
        mr=json.loads((folder/'map/report.json').read_text())
        assert sha256_file(folder/'map/map.pt')==mr['map_sha256']
        data=torch.load(folder/'map/map.pt',map_location='cuda',weights_only=False)
        fieldroot=Path(mr['root'])
        gp=fieldroot.parent/'frozen_geometry/report.json'
        gr=json.loads(gp.read_text())
        assert int(target['canonical_index']) not in gr['mapping_frames']
        geo=FrozenGaussianGeometry.load(gr['results']['independent_scale']['geometry'],
            expected_sha256=gr['results']['independent_scale']['geometry_sha256'],device='cuda')
        geo.assert_binding(data['geometry_binding'])
        prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
        qrpath=Path('output/nvos_positive_queries_20260918')/name/'receipt.json'
        qr=json.loads(qrpath.read_text())
        assert qr['negative_scribble_opened'] is False and qr['input_contract']=='nvos-ludvig-online-all-view-eight-v1'
        assert qr['reference_frame']==scene['prompt']['frame_id']
        assert qr['official_scribbles']['positive']['sha256']==manifest['protocol']['prompt_asset_sha256'][name]['positive']
        crop=qr['image_query_crop']
        assert sha256_file(crop['path'])==crop['sha256']
        identity=json.loads((fieldroot/'native_query_encoder.json').read_text())
        if encoder is None:
            encoder=NativeThreeQueryEncoder(identity,'cuda:0')
            encoder_identity=identity
        assert identity==encoder_identity
        with torch.no_grad():q=F.normalize(encoder.encode_image(Image.open(crop['path']).convert('RGB')),dim=-1)
        out=folder/('ranking_evaluation_'+name)
        out.mkdir(exist_ok=False)
        scores={arm:F.normalize(data['features'][:,:d],dim=-1)@(q@data['basis'][:d].T).T
                for arm,d in [('native64',64),('native_residual',len(data['basis']))]}
        torch.save(dict(scores={k:v.cpu() for k,v in scores.items()},query=q.cpu(),map_sha256=mr['map_sha256']),out/'element_scores.pt')
        width,height=qr['prompt_raster_wh']
        camera=prior.cameras[int(target['canonical_index'])]
        k=camera.intrinsic.cuda().clone();k[0]*=width/camera.width;k[1]*=height/camera.height
        responses={}
        for arm,score in scores.items():
            colors=torch.cat([score*data['observed'][:,None],data['observed'].float()[:,None]],1)
            render,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),
                colors=colors,viewmats=camera.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,
                near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            mass=render[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            responses[arm]=torch.where(valid,render[0,...,0]/mass.clamp_min(1e-8),-1).cpu()
        torch.save(responses,out/'responses.pt')
        saved=True
        gt=np.asarray(Image.open(target['gt_mask_path']))
        gt=np.any(gt>127,axis=2) if gt.ndim==3 else gt>127
        metrics={arm:dict(ap=float(average_precision_score(gt.flatten(),cv2.resize(response.numpy(),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR).flatten()))) for arm,response in responses.items()}
        report=dict(scene=name,map_scene=mapname,diagnostic_only=True,final_prompt_completion_not_applied=True,
            final_iou_not_measured=True,metrics=metrics,negative_scribble_opened=False,target_rgb_opened=False,
            target_gt_after_prediction=True,independent_test=False,query_receipt_sha256=sha256_file(qrpath),
            map_report_sha256=sha256_file(folder/'map/report.json'),manifest_sha256=sha256_file(mp),
            gt_sha256=sha256_file(target['gt_mask_path']),predictions_sha256=sha256_file(out/'responses.pt'),
            implementation_sha256=sha256_file(__file__))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,metrics,flush=True)


if __name__=='__main__':main()
