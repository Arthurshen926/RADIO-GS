"""Source-only group/descriptor diagnostic. Not a deployment feature bank."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 r=a.root;receipt=json.loads((r/'episodes.json').read_text());assert sha256_file(r/'episodes.pt')==receipt['sha256']
 data=torch.load(r/'episodes.pt',weights_only=False);info=json.loads((r/'teachers.json').read_text());assert sha256_file(r/'teachers.json')==data['teacher_manifest_sha256']
 parent_report=json.loads((r/a.parent/'report.json').read_text());parent=r/a.parent/'field.pt';assert sha256_file(parent)==parent_report['field_sha256'];field=torch.load(parent,weights_only=False)
 assert field['geometry_binding']==data['geometry_binding'] and field['query_encoder']==json.loads((r/'native_query_encoder.json').read_text())
 state=field['state_dict'];observed=state['observed'];z=state['latent'];d=state['native_decoder.weight'];bias=state['native_decoder.bias']
 assert not bias.any(),'projection control requires zero-bias native decoder'
 torch.testing.assert_close(d.T@d,torch.eye(d.shape[1]),atol=1e-4,rtol=1e-4)
 trainviews={info['frames'].index(f) for f in info['train_frames']}
 ids=torch.tensor([i for i,v in enumerate(data['proposal_views'].tolist()) if v in trainviews])
 memberships=(data['region_probability'][ids].float()>.7)&observed[None]
 keep=memberships.any(1);ids=ids[keep];memberships=memberships[keep]
 # Pool normalized decoded element directions, exactly the native cosine field.
 lengths=F.linear(z[:,:d.shape[1]],d,bias).norm(dim=-1).clamp_min(1e-8)
 mean_z=memberships.float()@(z[:,:d.shape[1]]/lengths[:,None])/memberships.sum(1)[:,None]
 region=F.normalize(data['queries']['image'][ids].float(),dim=-1)
 descriptors={'native_region':region,'projected_region':F.normalize((region@d)@d.T,dim=-1),'field_mean':F.normalize(mean_z@d.T,dim=-1)}
 result={'schema':'source_group_diagnostic.v1','diagnostic_only':True,'source_only':True,'geometry_binding':data['geometry_binding'],'query_encoder':field['query_encoder'],'parent_field_sha256':sha256_file(parent),'episode_sha256':receipt['sha256'],'observed':observed,'memberships':memberships,'source_regions':ids,'source_views':data['proposal_views'][ids],'train_frames':info['train_frames'],'descriptors':descriptors,'membership_threshold':.7,'implementation_sha256':sha256_file(__file__)}
 a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,a.output);a.output.with_suffix('.json').write_text(json.dumps({'sha256':sha256_file(a.output),'groups':len(ids),'elements':len(z),'source_only':True,'diagnostic_only':True,'parent':str(parent),'implementation_sha256':sha256_file(__file__)},indent=2)+'\n');print(a.output,len(ids),flush=True)
if __name__=='__main__':main()
