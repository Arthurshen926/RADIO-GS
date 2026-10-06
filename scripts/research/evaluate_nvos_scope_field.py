"""Evaluate a source-trained image field with the retained positive-only NVOS query."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization
from radio_gs.models.native_scope_dictionary import NativeScopeDictionary,attach_scope_text_readout
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from scripts.audits.predict_nvos_positive_ensemble_next32 import refine_positive
from scripts.audits.nvos_reference_completion_next19_runtime import complete_reference_query


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--base',default='output/nvos_joint_scope_next35');p.add_argument('--protect-native-rejection',action='store_true');p.add_argument('--prompt-readout',choices=['reference_component','graph_cut','reference_scopes'],default='reference_component');p.add_argument('--graph-strength',type=float,default=1.);p.add_argument('--graph-geometry',choices=['local','retained'],default='local');p.add_argument('--graph-evidence',choices=['learned','scopes'],default='learned');p.add_argument('--output-base');p.add_argument('--element-path');a=p.parse_args();torch.set_num_threads(4)
    key='nvos_'+('horns_center' if a.scene=='horns_left' else a.scene);folder=Path(a.base)/key;data=load(folder/'field.pt');parent=load(data['base_field']);root=Path(data['root']);out=(Path('output/nvos_native_admission_next35')/key if a.protect_native_rejection else folder)/('evaluation_'+a.scene);out.mkdir(parents=True,exist_ok=True)
    if a.output_base:
        out=Path(a.output_base)/key/('evaluation_'+a.scene);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text());scene=next(s for s in manifest['scenes'] if s['scene_id']==a.scene)
    target=next(f for f in scene['frames'] if f['frame_id'] in scene['evaluation_frame_ids']);reference=next(f for f in scene['frames'] if f['frame_id']==scene['prompt']['frame_id'])
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    retained_report=json.loads((Path('output/nvos_reference_completion_next19')/a.scene/'report.json').read_text())
    query_root=Path(retained_report['query_root'])
    qr=json.loads((query_root/a.scene/'receipt.json').read_text());width,height=qr['prompt_raster_wh']
    identity=json.loads((root/'native_query_encoder.json').read_text());encoder=NativeThreeQueryEncoder(identity,'cuda');query=encoder.encode_image(Image.open(qr['image_query_crop']['path']).convert('RGB'));encoder.vision=None
    def mask(path):
        x=np.asarray(Image.open(path));return torch.from_numpy(np.any(x>127,axis=2) if x.ndim==3 else x>127)
    positive=mask(qr['official_scribbles']['positive']['path']);sam=mask(qr['mask']);initial=load(Path('output/nvos_scope_init_next35')/key/'field.pt');predictions={};details={}
    arms=[('retained',None),('initial_dictionary',initial),('trained',data)]
    if data.get('scope_after_prompt',False):arms.append(('prompt_conditioned',data))
    for name,payload in arms:
        model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.requires_grad_(False);model.eval()
        if payload is not None:
            if 'scope_hypergraph' in payload:
                from radio_gs.querying.scope_hypergraph import attach_scope_hypergraph
                attach_scope_hypergraph(model,payload,query_task='reference_prompt')
            else:attach_scope_text_readout(model,NativeScopeDictionary(payload))
        with torch.no_grad():
            scores=model.retrieve(query,'image')['scores'];observed=model.observed_for('image')
            if name=='retained':native_scores=scores.clone()
            elif a.protect_native_rejection:
                # New semantics may refine a native positive. Only subsequent
                # authorized reference evidence may override a native rejection.
                scores=torch.where(native_scores<.5,torch.minimum(scores,native_scores),scores)
            base,_=refine_positive(scores,observed,geo,prior.cameras[int(reference['canonical_index'])],prior,width,height,positive,sam,'prompt_sam')
            if name=='prompt_conditioned':
                base=model.scope_graph.refine(base,query,model,'image')
                base,_=refine_positive(base,observed,geo,prior.cameras[int(reference['canonical_index'])],prior,width,height,positive,sam,'prompt_only')
            if name=='trained' and a.prompt_readout=='graph_cut':
                from radio_gs.querying.seeded_extent_cut import seeded_extent_cut
                scores,detail=seeded_extent_cut(base,observed,geo,prior.cameras[int(reference['canonical_index'])],prior,width,height,positive,sam,model,strength=a.graph_strength,geometry=a.graph_geometry,evidence=a.graph_evidence)
            else:
                scores,detail=complete_reference_query(base,observed,geo,prior.cameras[int(reference['canonical_index'])],prior,width,height,positive,sam,'reference_core_component',model,base_stage='prompt_sam')
            if name=='trained' and a.prompt_readout=='reference_scopes':
                from radio_gs.querying.reference_scope_completion import complete_reference_scopes
                scores,completion=complete_reference_scopes(scores,observed,geo,prior.cameras[int(reference['canonical_index'])],prior,width,height,positive,sam,model)
                detail=dict(parent=detail,scope_completion=completion)
            if name=='trained' and a.element_path:
                scores=load(a.element_path)['elements'].cuda();detail=dict(element_source=a.element_path,parent=detail)
            camera=prior.cameras[int(target['canonical_index'])];k=camera.intrinsic.cuda().clone();k[0]*=width/camera.width;k[1]*=height/camera.height
            image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=torch.cat([scores,observed.float()[:,None]],1),viewmats=camera.world_to_camera.cuda()[None],Ks=k[None],width=width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            mass=image[0,...,1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);response=torch.where(valid,image[0,...,0]/mass.clamp_min(1e-8),0).cpu()
        predictions[name]=response;details[name]=detail;torch.save(dict(response=response,elements=scores.cpu()),out/(name+'.pt'));print(a.scene,name,'prediction saved',flush=True)
    truth=mask(target['gt_mask_path']);metrics={}
    for name,response in predictions.items():
        response=torch.from_numpy(cv2.resize(response.numpy(),(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR));binary=response>=.5;union=(binary|truth).sum()
        metrics[name]=dict(iou=float((binary&truth).sum()/union) if union else float(manifest['protocol']['empty_union_value']),**binary_ranking_metrics(response.flatten(),truth.flatten()))
    (out/'evaluation.json').write_text(json.dumps(dict(scene=a.scene,metrics=metrics,details=details,query_root=str(query_root),native_rejection_protected=a.protect_native_rejection,target_rgb_used=False,negative_prompt_used=False,target_gt_after_predictions=True,query_recipe=a.prompt_readout,development=True),indent=2)+'\n');print(metrics,flush=True)


if __name__=='__main__':main()
