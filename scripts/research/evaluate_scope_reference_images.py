"""Replay the existing LERF reference-image development queries and scorer."""
import argparse
import json
from pathlib import Path
import torch
from gsplat import rasterization
from radio_gs.models.native_scope_dictionary import NativeScopeDictionary,attach_scope_text_readout
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.evaluation.lerf_common import load_labels,ground_truth_masks,binary_iou
from radio_gs.training.retrieval_metrics import binary_ranking_metrics


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--base',required=True);a=p.parse_args();torch.set_num_threads(4)
    out=Path(a.base)/a.key;data=load(out/'field.pt');root=Path(data['root']);parent=load(data['base_field']);scene=a.key[5:];directory=root.parent.parent/'evaluation'
    frozen=json.loads((directory/'reference_online.json').read_text());records=load(directory/'reference_online.encodings.pt');q=torch.cat([r['encoding'].half().float() for r in records if r['kind']=='image']).cuda()
    initial=load(Path('output/scope_query_modes_next34')/a.key/'field.pt');initial['native_modalities']=['text','image']
    model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.eval();support=model.observed_for('image')
    scores={'retained':model.retrieve(q,'image')['scores']}
    attach_scope_text_readout(model,NativeScopeDictionary(initial));scores['initial_dictionary']=model.retrieve(q,'image')['scores']
    model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(parent['state_dict']);model.eval()
    if 'source_kernel_extent' in data:
        from radio_gs.querying.source_kernel_extent import attach_source_kernel_extent
        attach_source_kernel_extent(model,data)
    elif 'region_native_features' in data:
        from radio_gs.querying.region_native_field import attach_region_native_field
        attach_region_native_field(model,data)
    elif 'scope_hypergraph' in data:
        from radio_gs.querying.scope_hypergraph import attach_scope_hypergraph
        attach_scope_hypergraph(model,data,query_task='image_category_union')
    else:attach_scope_text_readout(model,NativeScopeDictionary(data))
    scores['trained']=model.retrieve(q,'image')['scores']
    torch.save({k:v.cpu() for k,v in scores.items()},out/'reference_image_scores.pt')
    gr=json.loads(Path(data['geometry_report']).read_text());g=gr['results']['independent_scale'];geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda');prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256'])
    annotations,categories,sh,sw=load_labels(Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label'),scene);height=round(sh*480/sw)
    tasks=next(iter(frozen['runs'].values()))['observations'];results={k:[] for k in scores}
    colors=torch.cat(list(scores.values())+[support.float()[:,None]],dim=1);count=len(q)
    for frame in sorted({r['target_frame'] for r in tasks}):
        camera=prior.cameras[frame];k=camera.intrinsic.cuda().clone();k[0]*=480/camera.width;k[1]*=height/camera.height
        image,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,viewmats=camera.world_to_camera.cuda()[None],Ks=k[None],width=480,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
        mass=image[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8);targets=ground_truth_masks(annotations[frame],categories,height,480,sh,sw)
        for t in [r for r in tasks if r['target_frame']==frame]:
            truth=targets[t['category']].cuda()
            for i,name in enumerate(scores):
                response=torch.where(valid,image[0,...,i*count+t['reference']]/mass.clamp_min(1e-8),0)
                iou,_,_=binary_iou(response>.5,truth);results[name].append(dict(frame=frame,category=t['category'],iou=iou,**binary_ranking_metrics(response,truth)))
    macro={name:{metric:sum(r[metric] for r in rows)/len(rows) for metric in ['iou','average_precision','auroc']} for name,rows in results.items()}
    result=dict(key=a.key,macro=macro,observations=results,reference_contract=frozen['protocol']['query_contract'],development=True)
    (out/'reference_image_evaluation.json').write_text(json.dumps(result,indent=2)+'\n');print(macro,flush=True)


if __name__=='__main__':main()
