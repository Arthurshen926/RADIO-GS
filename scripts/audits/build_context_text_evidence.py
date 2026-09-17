"""Source-only crop-context agreement evidence; never reads benchmark labels."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--foreground-control',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
 info=json.loads((a.root/'teachers.json').read_text());receipt=json.loads((a.root/'episodes.json').read_text());source=json.loads((a.root/'source_frames.json').read_text())
 if sha256_file(a.root/'episodes.pt')!=receipt['sha256']:raise ValueError('episode identity')
 data=torch.load(a.root/'episodes.pt',weights_only=False)
 if sha256_file(a.root/'teachers.json')!=data['teacher_manifest_sha256']:raise ValueError('teacher identity')
 sigrecord=info['teacher_files']['siglip'];sigpath=a.root/sigrecord['path']
 if sha256_file(sigpath)!=sigrecord['sha256']:raise ValueError('SigLIP identity')
 sig=torch.load(sigpath,weights_only=False);identity=json.loads((a.root/'native_query_encoder.json').read_text())
 trainviews=[info['frames'].index(x) for x in info['train_frames']]
 regions=torch.where(torch.isin(data['proposal_views'],torch.tensor(trainviews)))[0]
 crops=[];records=[];sources={x['frame_idx']:x for x in source['frames']}
 for vi in trainviews:
  frame=info['frames'][vi];src=sources[frame];path=Path(source['image_dir'])/src['source_file']
  if sha256_file(path)!=src['source_sha256']:raise ValueError('source RGB identity')
  im=Image.open(path).convert('RGB');iw,ih=im.size
  for q in regions[data['proposal_views'][regions]==vi].tolist():
   mask=data['masks'][q];ys,xs=torch.where(mask);h,w=mask.shape
   left=float(xs.min())*iw/w;right=(float(xs.max())+1)*iw/w;top=float(ys.min())*ih/h;bottom=(float(ys.max())+1)*ih/h
   cx=(left+right)/2;cy=(top+bottom)/2;boxes=[]
   if a.foreground_control:
    erased=im.copy();native_mask=Image.fromarray((mask.numpy()*255).astype('uint8')).resize(im.size,Image.Resampling.NEAREST)
    erased.paste((127,127,127),(0,0,iw,ih),native_mask)
   for factor in (1.5,2.):
    cw=max((right-left)*factor,iw/3);ch=max((bottom-top)*factor,ih/3)
    box=(max(0,int(cx-cw/2)),max(0,int(cy-ch/2)),min(iw,int(cx+cw/2+1)),min(ih,int(cy+ch/2+1)))
    crops.append(pil_to_tensor(im.crop(box).resize((384,384),Image.Resampling.BICUBIC)).float()/255);boxes.append(box)
    if a.foreground_control:crops.append(pil_to_tensor(erased.crop(box).resize((384,384),Image.Resampling.BICUBIC)).float()/255)
   records.append({'region':q,'view':vi,'frame':frame,'source_sha256':src['source_sha256'],'boxes':boxes})
 model=NativeSiglip2Runtime.load(Path(identity['siglip_root']),device=torch.device(a.device));descriptors=[]
 for start in range(0,len(crops),8):
  descriptors.append(model.encode(torch.stack(crops[start:start+8])).cpu());print(start,len(crops),flush=True)
 descriptors=torch.cat(descriptors)
 background=None
 if a.foreground_control:
  paired=descriptors.reshape(len(records),2,2,-1);descriptors=paired[:,:,0];background=paired[:,:,1]
 else:descriptors=descriptors.reshape(len(records),2,-1)
 scores=descriptors@F.normalize(sig['text_embeddings'].float(),dim=-1).T
 top=scores.topk(2,dim=-1);labels=top.indices[:,:,0];margins=top.values[:,:,0]-top.values[:,:,1]
 accepted=(labels[:,0]==labels[:,1])&(margins.min(1).values>=.01)
 delta=None
 if a.foreground_control:
  background_scores=background@F.normalize(sig['text_embeddings'].float(),dim=-1).T
  delta=(scores-background_scores).gather(-1,labels[:,:,None])[:,:,0]
  accepted=accepted&(delta.min(1).values>=.01)
 original=(F.normalize(sig['descriptors'][[r['region'] for r in records]].float(),dim=-1)@F.normalize(sig['text_embeddings'].float(),dim=-1).T).argmax(1)
 for i,r in enumerate(records):r.update(foreground_evidence_delta=delta[i].tolist() if delta is not None else None,context_labels=[sig['queries'][x] for x in labels[i].tolist()],margins=margins[i].tolist(),accepted=bool(accepted[i]),original_label=sig['queries'][original[i]])
 payload={'schema':'source_context_text_evidence.v1','episode_sha256':receipt['sha256'],'teacher_manifest_sha256':data['teacher_manifest_sha256'],'geometry_binding':data['geometry_binding'],'native_encoder_identity':identity,'regions':torch.tensor([r['region'] for r in records]),'foreground_control':a.foreground_control,'foreground_evidence_delta':delta,'descriptors':descriptors.half(),'scores':scores,'labels':labels,'accepted':accepted,'records':records,'queries':sig['queries'],'policy':'two unmasked context scales1.5/2 with minimum crop extent1/3 frame; agree top1 and both margin>=.01; heuristic evidence, not calibrated truth'+('; both scales foreground-erasure class cosine drop>=.01' if a.foreground_control else ''),'training_frames_only':True}
 torch.save(payload,a.output)
 summary={k:payload[k] for k in ['schema','episode_sha256','policy','training_frames_only','records']};summary.update(sha256=sha256_file(a.output),regions=len(records),accepted=int(accepted.sum()),changed=int((accepted&(labels[:,0]!=original)).sum()))
 a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:summary[k] for k in ['regions','accepted','changed']})

if __name__=='__main__':main()
