"""Frozen current map: select 3D elements first, then render only that subset.

Implements the public LERF object-selection output order, but local polygon GT
is not certified equivalent to the released JPEG masks. Never claim volumetric
3D IoU. Selection is fixed once per query, independently of target view or GT.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.v4.evaluation.lerf_common import load_labels, ground_truth_masks, binary_iou
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.utils.query_input_identity import query_call_sha256
from radio_gs.scripts.evaluate_opengaussian_lerf_masks import FRAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry-report',type=Path,required=True);p.add_argument('--scene',choices=list(FRAMES),required=True);p.add_argument('--run',default='counterfactual_extent_benefit_admission');p.add_argument('--encoding-bank',type=Path,required=True);p.add_argument('--encoding-receipt',type=Path,required=True);p.add_argument('--label-root',type=Path,default=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'));p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    forbidden=('caption_region_evidence.pt','caption_grounded_regions.pt','caption_source_phrases.json','source_grids.pt','teacher_radio.pt','teacher_dino.pt','teacher_sam.pt','teacher_siglip.pt','episodes.pt','queries.pt','canonical_teacher_features','grounded_text_supervision','source_grounded_text','source_context_text')
    def deny_training(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in forbidden):raise RuntimeError('training asset access forbidden during selection evaluation')
    sys.addaudithook(deny_training)
    info=json.loads((a.root/'teachers.json').read_text());identity=json.loads((a.root/'native_query_encoder.json').read_text());auth=json.loads((a.root/'responsibility.json').read_text());gr=json.loads(a.geometry_report.read_text());g=gr['results']['independent_scale']
    assert sha256_file(a.root/'responsibility.json')==info['authority_sha256'];assert sha256_file(a.geometry_report)==auth['metadata']['frozen_geometry_report_sha256']
    fp=a.root/a.run/'field.pt';digest=json.loads(fp.with_name('report.json').read_text())['field_sha256'];assert sha256_file(fp)==digest
    geometry=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device);prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    annotations,categories,height,width=load_labels(a.label_root,a.scene)
    assert set(annotations)==set(FRAMES[a.scene]);assert set(annotations).isdisjoint(info['train_frames'])
    receipt=json.loads(a.encoding_receipt.read_text());assert receipt['encoding_bank_sha256']==sha256_file(a.encoding_bank)
    bank=torch.load(a.encoding_bank,map_location='cpu',weights_only=False);assert len(bank)==1 and bank[0]['kind']=='text';row=bank[0]
    assert row['texts']==categories and row['query_input_sha256']==query_call_sha256('text',(categories,),{},identity)
    queries=row['encoding'].half().float().to(a.device)
    field,_=load_element_retrieval_field(fp,expected_sha256=digest,geometry=geometry,expected_query_encoder=identity,device=a.device)
    protocol=dict(schema='current_selected_lerf_polygon.v1',scene=a.scene,field_sha256=digest,geometry_sha256=g['geometry_sha256'],query_encoding_sha256=sha256_file(a.encoding_bank),query_encoding_receipt_sha256=sha256_file(a.encoding_receipt),source_only_model=True,training_asset_access_forbidden=True,independent_test=False,element_threshold=.5,pixel_rule='round(alpha*255) via floor(x+.5), clamp uint8, >10',rendering='only selected 3D elements; nonselected elements do not occlude; original geometry opacity/scale unchanged',selection_target_view_dependent=False,target_rgb_opened=False,threshold_search=False,gt='Local LERF category-union polygon masks at original annotation dimensions',released_jpeg_gt_equivalence_verified=False,official_scores_directly_comparable=False,volumetric_3d_iou=False,public_reference='https://github.com/yanmin-wu/OpenGaussian/blob/main/scripts/compute_lerf_iou.py',width=width,height=height,labels=[dict(path=str(f),sha256=sha256_file(f)) for f in sorted((a.label_root/a.scene).glob('frame_*.json'))],implementation_sha256=sha256_file(__file__))
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with torch.no_grad():
        response=field.retrieve(queries,'text')['scores'];selected=(response>.5)&field.observed[:,None]
        torch.save(dict(categories=categories,selected=selected.cpu(),field_sha256=digest,geometry_sha256=g['geometry_sha256']),a.output/'selection.pt')
        # Query selection is finished and immutable before target masks are used.
        observations=[]
        for frame,objects in sorted(annotations.items()):
            targets=ground_truth_masks(objects,categories,height,width,height,width)
            camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone();k[0]*=width/camera.width;k[1]*=height/camera.height
            for ci,category in enumerate(categories):
                if not targets[category].any():continue
                ids=selected[:,ci]
                if ids.any():
                    _,alpha,_=rasterization(means=geometry.get_xyz()[ids],quats=geometry.get_rotation()[ids],scales=geometry.get_scaling()[ids],opacities=geometry.get_opacity().reshape(-1)[ids],colors=torch.ones(int(ids.sum()),1,device=a.device),viewmats=camera.world_to_camera.to(a.device)[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
                    byte=(alpha[0,...,0]*255+.5).floor().clamp(0,255).to(torch.uint8).cpu()
                else:byte=torch.zeros(height,width,dtype=torch.uint8)
                path=a.output/f'frame_{frame:05d}_{category}.png';Image.fromarray(byte.numpy()).save(path)
                prediction=byte>10;target=targets[category].cpu();iou,intersection,union=binary_iou(prediction,target)
                observations.append(dict(frame=frame,category=category,selected_elements=int(ids.sum()),iou=iou,intersection=intersection,union=union,prediction_sha256=sha256_file(path)))
            print(a.scene,frame,'completed',flush=True)
    geometry.assert_binding(auth['metadata']['frozen_geometry_binding']);values=np.array([r['iou'] for r in observations]);assert len(values)>0
    result=dict(protocol=protocol,protocol_sha256=sha256_file(a.output/'protocol.json'),selection_sha256=sha256_file(a.output/'selection.pt'),observations=observations,tasks=len(values),mean_iou=float(values.mean()),acc_at_025=float((values>.25).mean()),acc_at_05=float((values>.5).mean()),geometry_unchanged=True)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(a.scene,result['mean_iou'],flush=True)

if __name__=='__main__':main()
