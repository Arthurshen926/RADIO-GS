"""Restore native SAM boundaries before source inverse rendering at 192x256.

Descriptors, source frames, correspondence and geometry are unchanged. Appended
coarse grounded masks lack native rasters and remain explicitly interpolated.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.scripts.build_frozen_geometry_authority import build
from radio_gs.training.source_sam_evidence import read_manifest,load_source_sam
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.utils.mask_resampling import resize_boolean_masks_nearest_exact
from radio_gs.training.small_region_inventory import discarded_small_region_ids
from radio_gs.utils.immutable_artifacts import sha256_file


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);key='lerf_'+a.scene;data=load(Path('output/rendered_scopes_next37')/key/'field.pt');root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());ep=load(root/'episodes.pt');old=load(root.parent.parent/'base/raw_field/teacher_sam.pt');source=json.loads((root/'source_frames.json').read_text());frames={r['frame_idx']:r for r in source['frames']}
 manifest=root.parent.parent/'base/source_sam/manifest.json';document,records=read_manifest(manifest)
 extra=Path('output/scope_hierarchy_full_next33')/key/'small_teacher.pt'
 if not extra.exists():extra=Path('output/small_region_restoration_next32')/key/'small_teacher.pt'
 small=load(extra);coarse=torch.cat([ep['masks'],small['masks']]);target=resize_boolean_masks_nearest_exact(coarse,(192,256));native_count=0;small_offset=len(ep['masks'])
 for vi,frame in enumerate(info['frames']):
  rec=frames[frame];image=Path(source['image_dir'])/rec['source_file'];sam,_=load_source_sam(document,records,image,rec['source_sha256']);native=torch.from_numpy(unpack_masks(sam['packed_masks'],sam['mask_shape'][1]).copy());high=resize_boolean_masks_nearest_exact(native,(192,256));low=resize_boolean_masks_nearest_exact(native,tuple(ep['masks'].shape[-2:]));ids=torch.where(old['view_indices']==vi)[0]
  # Component inventory construction retains parent rows as an unchanged prefix.
  target[ids]=high[old['native_proposal_indices'][ids]];native_count+=len(ids)
  small_ids=discarded_small_region_ids(low);target[small_offset:small_offset+len(small_ids)]=high[small_ids];small_offset+=len(small_ids);native_count+=len(small_ids)
  print(key,'restored native boundary',frame,flush=True)
 out=Path('output/high_resolution_scope_evidence_next38')/key;out.mkdir(parents=True,exist_ok=True);torch.save(dict(masks=target,native_regions=native_count,interpolated_regions=len(target)-native_count),out/'masks.pt')
 geometry=Path(data['geometry_report']);build(SimpleNamespace(geometry_report=str(geometry),expected_report_sha256=sha256_file(geometry),frame_ids=','.join(map(str,info['frames'])),height=192,width=256,device='cuda',output_dir=str(out/'authority')))
 report=dict(scene=a.scene,source_frames=info['frames'],resolution=[192,256],native_regions=native_count,interpolated_regions=len(target)-native_count,geometry_frozen=True,new_rgb_access=False);(out/'evidence.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
