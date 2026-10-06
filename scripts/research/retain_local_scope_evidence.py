"""Retain unmatched source proposals as local evidence, not matched objects."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--source',default='output/rendered_scopes_next37');p.add_argument('--output',default='output/local_rendered_scopes_next37');a=p.parse_args();torch.set_num_threads(4)
 source=Path(a.source)/a.key;data=load(source/'field.pt');graph=data['scope_hypergraph'];report=json.loads((source/'construction.json').read_text());root=Path(data['root']);info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text());ep=load(root/'episodes.pt');masks=ep['masks'];views=ep['proposal_views'];native=load(root/info['teacher_files']['siglip']['path'])['descriptors'].float()
 extra=Path('output/scope_hierarchy_full_next33')/a.key/'small_teacher.pt'
 if not extra.exists():extra=Path('output/small_region_restoration_next32')/a.key/'small_teacher.pt'
 if extra.exists():
  x=load(extra);masks=torch.cat([masks,x['masks']]);views=torch.cat([views,x['proposal_views']]);native=torch.cat([native,x['descriptors'].float()])
 represented={i for r in report['records'] for i in r['regions']};n,h=graph['shape'];rows=[graph['indices'][0]];cols=[graph['indices'][1]];weights=[graph['weights']];prototypes=[graph['prototypes']];added=0;pixels=masks.shape[-2]*masks.shape[-1]
 for vi in range(len(info['frames'])):
  ids=torch.tensor([int(i) for i in torch.where(views==vi)[0] if int(i) not in represented],dtype=torch.long)
  if not len(ids):continue
  hit=load(root/auth['views'][vi]['relative_path']);pid=hit['pixel_ids'].long().cuda();gid=hit['gaussian_ids'].long().cuda();weight=hit['base_weights'].float().cuda();mass=torch.zeros(pixels,device='cuda').scatter_add_(0,pid,weight)
  op=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce();adjoint=op.transpose(0,1).coalesce();den=torch.sparse.sum(op,dim=0).to_dense()[:,None]
  for start in range(0,len(ids),32):
   region=ids[start:start+32];target=masks[region].flatten(1).float().T.cuda();value=torch.sparse.mm(adjoint,target)/den.clamp_min(1e-8)
   # The local mask supervises its source view only. No absent other view
   # supplies negatives and no physical cross-view identity is asserted.
   for _ in range(40):value.add_(.95*torch.sparse.mm(adjoint,target-torch.sparse.mm(op,value))/den.clamp_min(1e-8)).clamp_(0,1)
   for j,ri in enumerate(region.tolist()):
    selected=torch.where(value[:,j]>1e-5)[0]
    if not len(selected):continue
    rows.append(selected.cpu());cols.append(torch.full((len(selected),),h+added,dtype=torch.long));weights.append(value[selected,j].cpu());prototypes.append(F.normalize(native[ri:ri+1],dim=1).half());report['records'].append(dict(regions=[ri],views=[vi],validation_views=[],validation_iou=None,elements=len(selected),kind='unmatched_local',identity_verified=False));added+=1
  print(a.key,'local source view',vi,'retained',added,flush=True)
 graph.update(indices=torch.stack([torch.cat(rows),torch.cat(cols)]),weights=torch.cat(weights),prototypes=torch.cat(prototypes),quality=torch.cat([graph['quality'],torch.zeros(added)]),shape=[n,h+added]);report['retained_local_proposals']=added;data['method']='inverse-rendered matched scopes and retained single-view evidence';out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True);torch.save(data,out/'field.pt');(out/'construction.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
