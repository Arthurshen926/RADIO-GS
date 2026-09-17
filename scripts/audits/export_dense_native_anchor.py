"""Source-only dense native teacher control with the parent's fixed SVD basis.

Frozen structure, geometry, valid support and query encoder remain identical.
Fine local RGB crops test evidence missing from the region-only anchor.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.pipeline_contracts import validate_contributions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',required=True);parser.add_argument('--device',required=True)
    a=parser.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'scale_affinity_v1_direct_control';out=root/'dense_local_native_control'
    out.mkdir(exist_ok=False)
    info=json.loads((root/'teachers.json').read_text());source=json.loads((root/'source_frames.json').read_text())
    auth=json.loads((root/'responsibility.json').read_text())
    assert sha256_file(root/'responsibility.json')==info['authority_sha256']
    report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False)
    identity=json.loads((root/'native_query_encoder.json').read_text());assert identity==payload['query_encoder']
    runtime=NativeSiglip2Runtime.load(Path(identity['siglip_root']),device=torch.device(a.device))
    basis=payload['state_dict']['native_decoder.weight'].to(a.device)
    assert not payload['state_dict']['native_decoder.bias'].any()
    height,width=auth['metadata']['feature_height'],auth['metadata']['feature_width']
    n=len(payload['state_dict']['latent']);numerator=torch.zeros(n,basis.shape[1],device=a.device)
    mass=torch.zeros(n,device=a.device);records=[];grids=[]
    sources={s['frame_idx']:s for s in source['frames']}
    with torch.inference_mode():
        for frame in info['train_frames']:
            entry=sources[frame];path=Path(source['image_dir'])/entry['source_file']
            assert sha256_file(path)==entry['source_sha256']
            image=Image.open(path).convert('RGB');crops=[]
            for y in range(8):
                for x in range(12):
                    for scale in [.125,.25]:
                        cx=(x+.5)*image.width/12;cy=(y+.5)*image.height/8
                        cw,ch=round(scale*image.width),round(scale*image.height)
                        left=min(max(0,round(cx-cw/2)),image.width-cw)
                        top=min(max(0,round(cy-ch/2)),image.height-ch)
                        crop=image.crop((left,top,left+cw,top+ch)).resize((384,384),Image.Resampling.BICUBIC)
                        crops.append(pil_to_tensor(crop).float()/255)
            encoded=torch.cat([runtime.encode(torch.stack(crops[i:i+8])).float() for i in range(0,len(crops),8)])
            native=F.normalize(encoded.reshape(96,2,-1).mean(1),dim=-1)
            compressed=native@basis
            grids.append(compressed.cpu())
            pixels=F.interpolate(compressed.reshape(8,12,-1).permute(2,0,1)[None],size=(height,width),mode='nearest')[0].flatten(1).T
            vi=info['frames'].index(frame);record=auth['views'][vi];hitpath=root/record['relative_path']
            assert sha256_file(hitpath)==record['sha256']
            hit=torch.load(hitpath,map_location=a.device,weights_only=False)
            pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float()
            validate_contributions(pid,gid,weight,num_pixels=height*width,num_elements=n)
            mapping=torch.sparse_coo_tensor(torch.stack([gid,pid]),weight,(n,height*width)).coalesce()
            numerator+=torch.sparse.mm(mapping,pixels);mass.scatter_add_(0,gid,weight)
            records.append(dict(frame=frame,rgb_sha256=entry['source_sha256'],contributions_sha256=record['sha256']))
            print(a.scene,frame,'encoded',len(crops),flush=True)
        state={k:v.clone() for k,v in payload['state_dict'].items()}
        observed=state['observed'].to(a.device);valid=observed&(mass>1e-8)
        state['latent'][valid.cpu(),:basis.shape[1]]=(numerator[valid]/mass[valid,None]).cpu()
        assert torch.equal(state['latent'][:,basis.shape[1]:],payload['state_dict']['latent'][:,basis.shape[1]:])
        torch.save(dict(payload,state_dict=state),out/'field.pt')
        torch.save(dict(source_only=True,not_deployment=True,frames=info['train_frames'],compressed_grids=torch.stack(grids)),out/'source_grids.pt')
    record=dict(report,schema='dense_local_native_anchor_control.v1',field_sha256=sha256_file(out/'field.pt'),
                parent_field_sha256=report['field_sha256'],source_records=records,source_grids_sha256=sha256_file(out/'source_grids.pt'),
                grid=[8,12],crop_scales=[.125,.25],source_only=True,benchmark_data_used=False,
                native_prefix_exactly_preserved=False,native_text_image_max_error=None,
                structural_state_exactly_preserved=True,prompt_response_unchanged=True,
                basis_unchanged=True,observed_unchanged=True,inherited_calibration=True,
                updated_elements=int(valid.sum()),fallback_elements=int((observed&~valid).sum()),
                implementation_sha256=sha256_file(__file__))
    (out/'report.json').write_text(json.dumps(record,indent=2)+'\n');print(out,flush=True)


if __name__=='__main__':main()
