"""Source-only, query-independent probes; no benchmark labels or text targets."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from PIL import Image,ImageDraw
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions


@torch.inference_mode()
def main():
    torch.set_num_threads(4);out=Path('output/same_map_queries/probes');out.mkdir(parents=True,exist_ok=True)
    original='configs/consolidated/ramen.scene.json';scene=json.loads(Path(original).read_text());native=torch.load(scene['native_field'],map_location='cpu',weights_only=False);authority=json.loads(Path(scene['source_rgb_authority']).read_text());paths={int(r['image_id'].split('_')[-1]):r['path'] for r in authority['images']}
    binding=native['geometry'];geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);frame=scene['source_frames'][len(scene['source_frames'])//2]
    bank=torch.load(Path(scene['native_region_directories'][0])/f'frame_{frame:05d}.pt',map_location='cpu',weights_only=False);sam=torch.load(bank['sam_path'],weights_only=False);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])[bank['native_proposal_indices'].numpy()]
    chosen=[]
    for index in np.argsort(-masks.sum((1,2))):
        mask=masks[index]
        if all((mask&prev).sum()/max(1,(mask|prev).sum())<.1 for prev in chosen):chosen.append(mask)
        if len(chosen)==2:break
    h,w=192,256;cam=gp.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
    renderer=SimpleNamespace(scaled_intrinsics=lambda width,height:k,near_plane=.001*gp.scale,far_plane=100*gp.scale)
    hit=rasterize_single_view_contributions(geo,renderer,cam.world_to_camera.cuda(),height=h,width=w,include_depths=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['weights'].float();mass=torch.zeros(len(geo.xyz),device='cuda').scatter_add_(0,gid,weight)
    config=dict(scene='ramen',geometry=binding,native_field=scene['native_field'],text_encoder=scene['text_encoder'],native_region_directories=scene['native_region_directories'],camera_report=scene['camera_report'],camera_report_sha256=scene['camera_report_sha256'],source_frames=scene['source_frames'],tracking_frames=scene['source_frames'],source_rgb={str(f):paths[f] for f in scene['source_frames']},source_fit_hw=[192,256],boundary_radius=1,checkpoint='checkpoints/sam3_modelscope/sam3.pt')
    (out/'scene.json').write_text(json.dumps(config,indent=2)+'\n');records=[];rgb=Image.open(paths[frame]).convert('RGB')
    for i,mask in enumerate(chosen):
        small=F.interpolate(torch.from_numpy(mask).float()[None,None].cuda(),size=(h,w),mode='nearest')[0,0];inside=torch.zeros_like(mass).scatter_add_(0,gid,weight*small.flatten()[pid]);purity=inside/mass.clamp_min(1e-8);score=inside*(purity>.9)*native['supported'].flatten().cuda();anchor=int(score.argmax())
        if float(score[anchor])<=0:raise ValueError('No sufficiently supported 3D probe anchor')
        ys,xs=np.where(mask);box=[max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(rgb.width,int(xs.max())+4),min(rgb.height,int(ys.max())+4)];crop=out/f'probe_{i}_image.png';rgb.crop(box).save(crop);Image.fromarray(mask.astype('uint8')*255).save(out/f'probe_{i}_source_mask.png')
        image_query=dict(id=f'probe_{i}_image',modality='image',intent='reference_instance',image=str(crop),exclude_retrieval_frames=[frame])
        crop_mask=out/f'probe_{i}_crop_mask.png';Image.fromarray(mask.astype('uint8')*255).crop(box).save(crop_mask)
        masked_query=dict(image_query,id=f'probe_{i}_image_masked',foreground_mask=str(crop_mask),input_contract='RGB plus explicit foreground mask; stronger than RGB-only')
        point_query=dict(id=f'probe_{i}_point3d',modality='prompt_3d',intent='reference_instance',world_xyz=geo.xyz[anchor].cpu().tolist(),geometry=binding)
        for name,q in [('image',image_query),('image_masked',masked_query),('point3d',point_query)]:(out/f'probe_{i}_{name}.json').write_text(json.dumps(q,indent=2)+'\n')
        records.append(dict(probe=i,source_frame=frame,crop_box=box,anchor_gaussian=anchor,mask=str(out/f'probe_{i}_source_mask.png'),image_query=str(out/f'probe_{i}_image.json'),point_query=str(out/f'probe_{i}_point3d.json')))
    (out/'probes.json').write_text(json.dumps(dict(policy='two largest mutually IoU<0.1 cached native regions in middle authorized source frame; anchors >0.9 region footprint purity',independent_ground_truth=False,benchmark_queries_used=False,records=records),indent=2)+'\n')


if __name__=='__main__':main()
