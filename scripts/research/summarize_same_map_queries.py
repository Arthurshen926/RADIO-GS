"""Source-region diagnostics and coordinate-equivalence, NOT benchmark accuracy."""
import json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
    torch.set_num_threads(4);root=Path('output/same_map_queries');scene=json.loads((root/'probes/scene.json').read_text());probes=json.loads((root/'probes/probes.json').read_text())['records'];b=scene['geometry'];geo=FrozenGaussianGeometry.load(b['path'],expected_sha256=b['sha256'],device='cuda');gp=MogeGeometryPrior(scene['camera_report'],scene['camera_report_sha256']);rows=[];canvas=Image.new('RGB',(4*384,2*310),'white');draw=ImageDraw.Draw(canvas)
    for row,probe in enumerate(probes):
        i=probe['probe'];frame=probe['source_frame'];cam=gp.cameras[frame];h,w=288,384;k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
        reference=np.asarray(Image.open(probe['mask']).convert('L').resize((w,h),Image.Resampling.NEAREST))>127;rgb=np.asarray(Image.open(scene['source_rgb'][str(frame)]).convert('RGB').resize((w,h))).copy()
        modes={};sets={};heads={}
        for mode in ['image','image_masked','point3d','point2d']:
            dest=root/f'probe_{i}_{mode}';d=load(dest/'runtime/typed_readout.pt');member=(d['selection_scores'][:,0]>.5)&d['selection_observed'];sets[mode]=member;heads[mode]=d
            member=member.cuda();_,alpha,_=rasterization(means=geo.xyz[member],quats=geo.quaternions[member],scales=geo.scales[member],opacities=geo.opacities.flatten()[member],colors=torch.ones(int(member.sum()),1,device='cuda'),viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*gp.scale,far_plane=100*gp.scale,packed=False)
            mask=alpha[0,...,0].cpu().numpy()>10.5/255
            modes[mode]=dict(selected_elements=int(member.sum()),source_region_iou=float((mask&reference).sum()/max(1,(mask|reference).sum())),geometry=d['selection_geometry'])
            if mode!='point2d':
                col={'image':1,'image_masked':2,'point3d':3}[mode];overlay=rgb.copy();overlay[mask]=(overlay[mask]*.45+np.array([230,50,30])*.55).astype('uint8');canvas.paste(Image.fromarray(overlay),(col*w,row*310+22));draw.text((col*w+4,row*310+4),f"{mode}: source IoU {modes[mode]['source_region_iou']:.3f}",fill='black')
        view=rgb.copy();view[reference]=(view[reference]*.45+np.array([30,190,90])*.55).astype('uint8');canvas.paste(Image.fromarray(view),(0,row*310+22));draw.text((4,row*310+4),f'probe {i}: source pseudo-region (not GT)',fill='black')
        a=sets['point2d'];bset=sets['point3d'];equivalence=float((a&bset).sum()/(a|bset).sum().clamp_min(1));response_diff=float((heads['point2d']['scores']-heads['point3d']['scores']).abs().max())
        for mode in ['image','image_masked']:
            base=root/f'probe_{i}_{mode}';resolved=base/'resolved' if (base/'resolved').exists() else base/'runtime/resolved';native=load(resolved/'native_response.pt');v=native['cosine'][:,0];known=native['observed'].reshape(-1);anchor=v[probe['anchor_gaussian']]
            modes[mode]['anchor_top_fraction']=float(((v[known]>anchor).sum()+1)/known.sum());modes[mode]['resolution']=json.loads((resolved/'resolution.json').read_text())['details']
        rows.append(dict(probe=i,source_frame=frame,modes=modes,point_coordinate_control=dict(set_iou=equivalence,max_response_difference=response_diff),independent_ground_truth=False))
    canvas.save(root/'source_comparison.jpg');(root/'summary.json').write_text(json.dumps(dict(probes=rows,official_benchmark=False,source_masks_are_pseudolabels=True,point2d_derived_from_point3d=True,origin_frame_excluded_from_image_retrieval_only=True,shared_geometry=scene['geometry']),indent=2)+'\n')
    for r in rows:print(r['probe'],{m:round(v['source_region_iou'],4) for m,v in r['modes'].items()},r['point_coordinate_control'])


if __name__=='__main__':main()
