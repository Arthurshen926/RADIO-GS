"""Real LERF polygons, native text queries, frozen full-scene feature rendering.

Development protocol, not the historical selected-only or official 3D protocol.
No benchmark RGB, target labels, or training assets enter query encoding.
"""
import argparse
import json
import sys
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field, retrieve_contrastive_text
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.v4.evaluation.lerf_common import load_labels, ground_truth_masks, binary_iou
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.retrieval_metrics import binary_ranking_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--geometry-report',type=Path,required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--label-root',type=Path,default=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'))
    p.add_argument('--width',type=int,default=480)
    p.add_argument('--device',default='cuda:4')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview-dir',type=Path)
    p.add_argument('--generic-negative-readout',action='store_true')
    p.add_argument('--text-template',choices=['verbatim','source_photo'],default='verbatim')
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists() or a.output.with_suffix('.protocol.json').exists():raise FileExistsError(a.output)
    if a.preview_dir is not None:a.preview_dir.mkdir(parents=True,exist_ok=False)
    if a.width<1:raise ValueError('invalid raster width')
    def deny_training(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ('caption_region_evidence.pt','caption_grounded_regions.pt','caption_source_phrases.json','source_grids.pt','teacher_radio.pt','teacher_dino.pt','teacher_sam.pt','teacher_siglip.pt','episodes.pt','queries.pt','canonical_teacher_features','grounded_text_supervision','source_grounded_text','source_context_text','consistent_native_element_teacher')):
            raise RuntimeError('training asset access forbidden')
    sys.addaudithook(deny_training)
    info=json.loads((a.root/'teachers.json').read_text())
    identity=json.loads((a.root/'native_query_encoder.json').read_text())
    report=json.loads(a.geometry_report.read_text());g=report['results']['independent_scale']
    authority=json.loads((a.root/'responsibility.json').read_text())
    if sha256_file(a.root/'responsibility.json')!=info['authority_sha256']:raise ValueError('authority mismatch')
    if sha256_file(a.geometry_report)!=authority['metadata']['frozen_geometry_report_sha256']:raise ValueError('geometry report mismatch')
    geometry=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
    prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
    run_records={}
    for run in a.runs:
        r=json.loads((a.root/run/'report.json').read_text())
        if sha256_file(a.root/run/'field.pt')!=r['field_sha256']:raise ValueError('field mismatch')
        run_records[run]=r['field_sha256']
    label_records=[{'path':str(x),'sha256':sha256_file(x)} for x in sorted((a.label_root/a.scene).glob('frame_*.json'))]
    if not label_records:raise ValueError('missing annotations')
    protocol={'schema':'frozen_field_lerf_polygon_development.v1','scene':a.scene,'width':a.width,'threshold':.5,
              'rendering':'full frozen scene; observed response mass / observed alpha; require observed/full alpha > .5 and full alpha > 1e-8',
              'queries':'native category text; target RGB and threshold search absent; readout policy explicit','text_template':a.text_template,'readout':'generic_alternative_margin' if a.generic_negative_readout else 'absolute_membership','generic_alternatives':['object','things','stuff','texture'] if a.generic_negative_readout else [],
              'labels':label_records,'geometry_sha256':g['geometry_sha256'],'fields':run_records,
              'semantic_train_frames':info['train_frames'],'benchmark_ground_truth':True,'official_benchmark_protocol':False,
              'historical_selected_only_scores_directly_comparable':False,'training_asset_access_forbidden':True,
              'implementation_sha256':sha256_file(__file__)}
    a.output.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    annotations,categories,sh,sw=load_labels(a.label_root,a.scene)
    if set(annotations)&set(info['train_frames']):raise ValueError('annotation view overlaps semantic training')
    if not set(annotations)<=set(prior.cameras):raise ValueError('missing annotation cameras')
    encoder=NativeThreeQueryEncoder(identity,a.device)
    phrases=categories if a.text_template=='verbatim' else ['a photo of a '+c for c in categories]
    queries=encoder.encode_text(phrases).half().float()
    negatives=None
    if a.generic_negative_readout:
        alternatives=['object','things','stuff','texture']
        if a.text_template=='source_photo':alternatives=['a photo of a '+x for x in alternatives]
        negatives=encoder.encode_text(alternatives).half().float()
    height=round(sh*a.width/sw)
    results={}
    with torch.no_grad():
        for run,digest in run_records.items():
            field,_=load_element_retrieval_field(a.root/run/'field.pt',expected_sha256=digest,geometry=geometry,expected_query_encoder=identity,device=a.device)
            response=(field.retrieve(queries,'text') if negatives is None else retrieve_contrastive_text(field,queries,negatives))['scores']
            colors=torch.cat([response,field.observed.float()[:,None]],dim=1)
            rows=[]
            for frame,objects in sorted(annotations.items()):
                camera=prior.cameras[frame];k=camera.intrinsic.to(a.device).clone()
                k[0]*=a.width/camera.width;k[1]*=height/camera.height
                image,alpha,_=rasterization(means=geometry.get_xyz(),quats=geometry.get_rotation(),scales=geometry.get_scaling(),
                    opacities=geometry.get_opacity().reshape(-1),colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],
                    Ks=k[None],width=a.width,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
                mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
                rendered=image[0,...,:-1]/mass[...,None].clamp_min(1e-8)
                targets=ground_truth_masks(objects,categories,height,a.width,sh,sw)
                if a.preview_dir is not None and frame==min(annotations):
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    selected=[(i,c) for i,c in enumerate(categories) if targets[c].any()][:3]
                    fig,axes=plt.subplots(len(selected),3,figsize=(10,3*len(selected)),squeeze=False)
                    for ri,(ci,c) in enumerate(selected):
                        prediction=((rendered[...,ci]>.5)&valid).cpu()
                        for col,img,title in [(0,targets[c],c+' / GT'),(1,rendered[...,ci].cpu(), 'response'),(2,prediction,'prediction > 0.5')]:
                            axes[ri,col].imshow(img,vmin=0,vmax=1,cmap='viridis' if col==1 else 'gray');axes[ri,col].set_title(title);axes[ri,col].axis('off')
                    fig.suptitle(a.scene+' / '+run+' / frame '+str(frame));fig.tight_layout();fig.savefig(a.preview_dir/(run+'.png'),dpi=130);plt.close(fig)
                for ci,category in enumerate(categories):
                    target=targets[category].to(a.device)
                    if not target.any():continue
                    pred=(rendered[...,ci]>.5)&valid
                    iou,intersection,union=binary_iou(pred,target)
                    ranking=binary_ranking_metrics(torch.where(valid,rendered[...,ci],torch.zeros_like(mass)),target)
                    rows.append({**ranking,'frame':frame,'category':category,'iou':iou,'intersection':intersection,'union':union,
                                 'positive_pixels':int(target.sum()),'false_positive_pixels':int((pred&~target).sum()),
                                 'positive_support_recall':float((target&valid).sum()/target.sum()),
                                 'supported_fraction':float(valid.float().mean())})
            if not rows:raise ValueError('no annotation tasks')
            macro={c:sum(r['iou'] for r in rows if r['category']==c)/sum(r['category']==c for r in rows) for c in categories}
            results[run]={'field_sha256':digest,'tasks':len(rows),'categories':len(categories),'observation_miou':sum(r['iou'] for r in rows)/len(rows),
                          'category_macro_miou':sum(macro.values())/len(macro),'positive_support_recall_macro':sum(r['positive_support_recall'] for r in rows)/len(rows),
                          'average_precision_macro':sum(r['average_precision'] for r in rows)/len(rows),'auroc_macro':sum(r['auroc'] for r in rows)/len(rows),'per_category_iou':macro,'observations':rows}
            print(run,results[run]['observation_miou'],results[run]['positive_support_recall_macro'],flush=True)
    geometry.assert_binding(authority['metadata']['frozen_geometry_binding'])
    a.output.write_text(json.dumps({'protocol':protocol,'protocol_sha256':sha256_file(a.output.with_suffix('.protocol.json')),'annotation_frames':sorted(annotations),'runs':results,'geometry_unchanged':True},indent=2)+'\n')

if __name__=='__main__':main()
