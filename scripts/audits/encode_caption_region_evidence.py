"""Preserve native SAM boundaries for teacher encoding and lift source memberships."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.training.region_visibility import normalize_visible_region_mass
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--root',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    out=root/'caption_region_evidence.pt'
    if out.exists():raise FileExistsError(out)
    path=root/'caption_grounded_regions.pt';receipt=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==receipt['sha256']
    g=torch.load(path,weights_only=False);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    assert g['teacher_manifest_sha256']==sha256_file(root/'teachers.json') and sha256_file(root/'responsibility.json')==info['authority_sha256']
    source=json.loads((root/'source_frames.json').read_text());sources={s['frame_idx']:s for s in source['frames']}
    identity=json.loads((root/'native_query_encoder.json').read_text());runtime=NativeSiglip2Runtime.load(Path(identity['siglip_root']),device=torch.device(a.device))
    h,w=g['masks'].shape[1:];n=auth['num_gaussians'];native_desc=[];coarse_desc=[];prob=torch.zeros(len(g['masks']),n,dtype=torch.float16)
    seen=torch.zeros(len(info['frames']),n,dtype=torch.bool)
    with torch.inference_mode():
        for frame in info['train_frames']:
            vi=info['frames'].index(frame);ids=torch.where(g['proposal_views']==vi)[0];src=sources[frame]
            imagepath=Path(source['image_dir'])/src['source_file'];assert sha256_file(imagepath)==src['source_sha256']
            image=Image.open(imagepath).convert('RGB');rgb=np.asarray(image);native_crops=[];coarse_crops=[]
            for index in ids.tolist():
                r=g['records'][index];assert r['frame']==frame and r['rgb_sha256']==src['source_sha256']
                native=np.unpackbits(r['packed_native'].numpy(),axis=-1)[:,:r['native_shape'][1]].astype(bool)
                coarse=np.asarray(Image.fromarray(g['masks'][index].numpy()).resize(image.size,Image.Resampling.NEAREST)).astype(bool)
                for mask,container in [(native,native_crops),(coarse,coarse_crops)]:
                    ys,xs=np.where(mask);masked=rgb.copy();masked[~mask]=127
                    box=(max(0,int(xs.min())-3),max(0,int(ys.min())-3),min(image.width,int(xs.max())+4),min(image.height,int(ys.max())+4))
                    container.append(pil_to_tensor(Image.fromarray(masked).crop(box).resize((384,384),Image.Resampling.BICUBIC)).float()/255)
            for crops,container in [(native_crops,native_desc),(coarse_crops,coarse_desc)]:
                container.append(torch.cat([runtime.encode(torch.stack(crops[start:start+8])).cpu().half() for start in range(0,len(crops),8)]))
            record=auth['views'][vi];hitpath=root/record['relative_path'];assert sha256_file(hitpath)==record['sha256']
            hit=torch.load(hitpath,map_location=a.device,weights_only=False);gid=hit['gaussian_ids'].long();pid=hit['pixel_ids'].long();weight=hit['base_weights'].float()
            pixelmass=torch.zeros(h*w,device=a.device).scatter_add_(0,pid,weight);normalized=weight/pixelmass[pid].clamp_min(1e-8)
            mapping=torch.sparse_coo_tensor(torch.stack([gid,pid]),normalized,(n,h*w)).coalesce();elementmass=torch.zeros(n,device=a.device).scatter_add_(0,gid,normalized)
            lifted,visible=normalize_visible_region_mass(torch.sparse.mm(mapping,g['masks'][ids].float().flatten(1).to(a.device).T),elementmass)
            prob[ids]=lifted.T.cpu().half();seen[vi]=visible.cpu()
            print(a.scene,frame,len(ids),'encoded/lifted',flush=True)
    del runtime
    encoder=NativeThreeQueryEncoder(identity,a.device);text=encoder.encode_text(g['phrases']).cpu().half()
    bundle=dict(schema='caption_region_evidence.v1',source_only=True,benchmark_data_used=False,grounding_sha256=receipt['sha256'],
        teacher_manifest_sha256=g['teacher_manifest_sha256'],geometry_binding=g['geometry_binding'],train_frames=info['train_frames'],
        masks=g['masks'],proposal_views=g['proposal_views'],region_probability=prob,seen=seen,
        native_descriptors=torch.cat(native_desc),coarse_descriptors=torch.cat(coarse_desc),
        records=[{k:v for k,v in r.items() if k!='packed_native'} for r in g['records']],phrases=g['phrases'],text_embeddings=text,
        implementation_sha256=sha256_file(__file__))
    torch.save(bundle,out);out.with_suffix('.json').write_text(json.dumps(dict(sha256=sha256_file(out),regions=len(prob),implementation_sha256=bundle['implementation_sha256']),indent=2)+'\n');print(out,flush=True)


if __name__=='__main__':main()
