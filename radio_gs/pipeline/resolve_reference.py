"""Resolve image identity or visible 3D anchors to a reference-instance query."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.pipeline.native_field import score_field


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
    scene=json.loads(Path(a.scene).read_text());query=json.loads(Path(a.query).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True);resolved=dict(query);detail={}
    if query['modality']=='image':
        from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
        runtime=NativeSiglip2Runtime.load(Path(scene['text_encoder']),device=torch.device('cuda'))
        crop=Image.open(query['image']).convert('RGB')
        if query.get('foreground_mask'):
            foreground=np.asarray(Image.open(query['foreground_mask']).convert('L'))>127
            rgb=np.array(crop,copy=True);rgb[~foreground]=127;crop=Image.fromarray(rgb)
        crop=crop.resize((384,384),Image.Resampling.BICUBIC);pixels=torch.from_numpy(np.array(crop,copy=True)).permute(2,0,1).float()[None]/255;encoding=F.normalize(runtime.encode(pixels).float(),dim=-1)
        field=torch.load(scene['native_field'],map_location='cpu',weights_only=False)
        cosine=score_field(field,encoding,'cuda');cosine[~field['supported'].reshape(-1)]=-torch.inf
        torch.save(dict(cosine=cosine,encoding=encoding.cpu(),geometry=field['geometry'],observed=field['supported'],calibration='rank-only; text probability calibration NOT used'),out/'native_response.pt')
        best=None
        for directory in scene['native_region_directories']:
            for frame in scene['source_frames']:
                if frame in query.get('exclude_retrieval_frames',[]):continue
                path=Path(directory)/f'frame_{frame:05d}.pt';native=torch.load(path,map_location='cpu',weights_only=False)
                if not len(native['descriptors']):continue
                sims=F.normalize(native['descriptors'].float(),dim=-1)@encoding.cpu().T;value,index=sims[:,0].max(0)
                if best is None or float(value)>best['score']:best=dict(path=str(path),frame=frame,index=int(index),score=float(value))
        native=torch.load(best['path'],map_location='cpu',weights_only=False);sam=torch.load(native['sam_path'],weights_only=False);mask=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[int(native['native_proposal_indices'][best['index']])];seed=out/'reference_mask.png';Image.fromarray(mask.astype('uint8')*255).save(seed)
        resolved.update(reference_frame=best['frame'],reference_rgb=scene['source_rgb'][str(best['frame'])],reference_mask=str(seed),reference_mask_is_gt=False,seed_origin='native image retrieval from declared source-region cache')
        detail=dict(seed=best,encoder=scene['text_encoder'],text_query_used=False,reference_origin_excluded=query.get('exclude_retrieval_frames',[]))
    elif query['modality']=='prompt_3d':
        if query['geometry']!=scene['geometry']:raise ValueError('3D point belongs to a different coordinate/geometry binding')
        b=scene['geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);xyz=torch.tensor(query['world_xyz'],device='cuda');distance=(geo.xyz-xyz).square().sum(1);anchor=int(distance.argmin());best=None;h,w=96,128
        for frame in scene['source_frames']:
            cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height;renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
            hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);is_anchor=hit['gaussian_ids']==anchor
            if not is_anchor.any():continue
            weights=hit['weights'][is_anchor];value,index=weights.max(0);pixel=int(hit['pixel_ids'][is_anchor][index])
            if best is None or float(value)>best['weight']:best=dict(frame=frame,pixel=pixel,weight=float(value))
        if best is None:raise ValueError('No authorized source view observes the clicked primitive')
        resolved.update(reference_frame=best['frame'],reference_rgb=scene['source_rgb'][str(best['frame'])],points_normalized=[[(best['pixel']%w+.5)/w,(best['pixel']//w+.5)/h]],point_labels=[1],seed_origin='maximum visible contribution pixel of nearest registered primitive')
        detail=dict(anchor_gaussian=anchor,anchor_distance=float(distance[anchor].sqrt()),view_choice=best,ray_raster=[h,w],world_point=query['world_xyz'])
    elif query['modality']=='prompt_2d':
        resolved['reference_rgb']=scene['source_rgb'][str(query['reference_frame'])]
    else:raise ValueError('Expected image, prompt_2d or prompt_3d')
    resolved['negative_prompt_used']=any(v==0 for v in resolved.get('point_labels',[]))
    (out/'query.json').write_text(json.dumps(resolved,indent=2)+'\n');(out/'resolution.json').write_text(json.dumps(dict(input_query=a.query,scene=a.scene,details=detail,benchmark_labels_used=False),indent=2)+'\n')


if __name__=='__main__':main()
