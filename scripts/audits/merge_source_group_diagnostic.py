"""Add conservative multi-view unions to a source-only diagnostic inventory."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.source_group_association import associate_source_groups
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--visibility-aware',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 receipt=json.loads(a.input.with_suffix('.json').read_text());assert sha256_file(a.input)==receipt['sha256'];pack=torch.load(a.input,weights_only=False)
 parent=Path(receipt['parent']);assert sha256_file(parent)==pack['parent_field_sha256'];state=torch.load(parent,weights_only=False)['state_dict']
 visibility=None
 if a.visibility_aware:
  path=parent.parent.parent/'episodes.pt';assert sha256_file(path)==pack['episode_sha256']
  data=torch.load(path,weights_only=False);assert data['geometry_binding']==pack['geometry_binding']
  visibility=data['seen'][pack['source_views']]
 association_masks=pack['memberships'] if visibility is None else pack['memberships']&visibility
 ids=torch.where(association_masks.any(1))[0]
 local=associate_source_groups(association_masks[ids],pack['descriptors']['native_region'][ids],pack['source_views'][ids],visibility=None if visibility is None else visibility[ids],minimum_shared_elements=20 if a.visibility_aware else 1)
 groups=[[int(ids[i]) for i in group] for group in local]
 masks=torch.stack([pack['memberships'][group].any(0) for group in groups]) if groups else pack['memberships'][:0]
 d=state['native_decoder.weight'];z=state['latent'][:,:d.shape[1]];length=F.linear(z,d,state['native_decoder.bias']).norm(dim=-1).clamp_min(1e-8)
 native=F.normalize(torch.stack([pack['descriptors']['native_region'][g].mean(0) for g in groups]),dim=-1) if groups else pack['descriptors']['native_region'][:0]
 mean=masks.float()@(z/length[:,None])/masks.sum(1)[:,None].clamp_min(1)
 extra={'native_region':native,'projected_region':F.normalize((native@d)@d.T,dim=-1),'field_mean':F.normalize(mean@d.T,dim=-1)}
 result=dict(pack,memberships=torch.cat([pack['memberships'],masks]),descriptors={k:torch.cat([v,extra[k]]) for k,v in pack['descriptors'].items()},association_groups=groups,original_groups=len(pack['memberships']),association_policy={'minimum_iou':.25,'minimum_cosine':.8,'minimum_size_ratio':.5,'mutual_best_per_view':True,'complete_link':True,'one_region_per_view':True,'visibility_aware':a.visibility_aware,'minimum_shared_elements':20 if a.visibility_aware else 1},source_candidate_sha256=receipt['sha256'],implementation_sha256=sha256_file(__file__),association_implementation_sha256=sha256_file(associate_source_groups.__code__.co_filename))
 torch.save(result,a.output);a.output.with_suffix('.json').write_text(json.dumps({'sha256':sha256_file(a.output),'parent':str(parent),'groups':len(result['memberships']),'added_groups':len(groups),'source_only':True,'diagnostic_only':True},indent=2)+'\n');print(a.output,len(groups),flush=True)
if __name__=='__main__':main()
