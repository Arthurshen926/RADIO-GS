"""Discover grounding phrases from training RGB only, without benchmark vocabulary."""
import argparse
import json
import re
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from radio_gs.utils.immutable_artifacts import sha256_file


def caption_phrases(caption):
    text=re.sub(r'[^a-z0-9 ,]',' ',caption.lower())
    text=re.sub(r'\b(?:a |an |the )?(?:close up|picture|photo|image|view) of\b',' ',text)
    chunks=re.split(r'\b(?:on top of|next to|in front of|filled with|topped with|covered in|on|at|in|with|and|is|are|sitting|standing|holding|against|behind)\b|,',text)
    phrases=set()
    for chunk in chunks:
        for part in [chunk,*chunk.split(' of ')]:
            part=re.sub(r'^\s*(?:a|an|the|some|there)\s+','',part).strip()
            part=' '.join(part.split())
            if 1<=len(part.split())<=6 and len(part)>2 and part not in {'background','white background','top','side','front','it','there','pair','lot','group','bunch'}:
                phrases.add(part)
    return sorted(phrases)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--root',type=Path)
    a=p.parse_args();torch.set_num_threads(4)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    out=root/'caption_source_phrases.json'
    if out.exists():raise FileExistsError(out)
    info=json.loads((root/'teachers.json').read_text());source=json.loads((root/'source_frames.json').read_text())
    rec=info['teacher_files']['sam'];assert sha256_file(root/rec['path'])==rec['sha256']
    sam=torch.load(root/rec['path'],weights_only=False);original=sam['source_receipts'][0].get('all_parent_regions_retained',len(sam['masks']))
    modelroot=next(Path('/root/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base/snapshots').iterdir())
    model_files={str(p.resolve(strict=True)):sha256_file(p.resolve(strict=True)) for p in modelroot.iterdir() if p.is_file()}
    processor=BlipProcessor.from_pretrained(modelroot,local_files_only=True)
    model=BlipForConditionalGeneration.from_pretrained(modelroot,local_files_only=True).to(a.device).eval().requires_grad_(False)
    records=[];sources={s['frame_idx']:s for s in source['frames']}
    with torch.inference_mode():
        for frame in info['train_frames']:
            src=sources[frame];path=Path(source['image_dir'])/src['source_file'];assert sha256_file(path)==src['source_sha256']
            im=Image.open(path).convert('RGB');images=[im];ids=[None];vi=info['frames'].index(frame)
            for i in range(original):
                if int(sam['view_indices'][i])!=vi:continue
                mask=sam['masks'][i];ys,xs=mask.nonzero(as_tuple=True)
                if not len(ys):continue
                x0=max(0,int(xs.min())*im.width//mask.shape[1]-3);y0=max(0,int(ys.min())*im.height//mask.shape[0]-3)
                x1=min(im.width,(int(xs.max())+1)*im.width//mask.shape[1]+3);y1=min(im.height,(int(ys.max())+1)*im.height//mask.shape[0]+3)
                images.append(im.crop((x0,y0,x1,y1)));ids.append(i)
            captions=[]
            for start in range(0,len(images),8):
                inputs=processor(images=images[start:start+8],return_tensors='pt').to(a.device)
                output=model.generate(**inputs,max_new_tokens=40,num_beams=3,do_sample=False)
                captions.extend(processor.batch_decode(output,skip_special_tokens=True))
            for index,caption in zip(ids,captions):records.append(dict(frame=frame,region=index,rgb_sha256=src['source_sha256'],caption=caption,phrases=caption_phrases(caption)))
            print(a.scene,frame,captions[0],flush=True)
    result=dict(schema='source_caption_phrases.v1',source_only=True,benchmark_vocabulary_used=False,
                train_frames=info['train_frames'],teacher_manifest_sha256=sha256_file(root/'teachers.json'),
                records=records,phrases=sorted({p for r in records for p in r['phrases']}),
                implementation_sha256=sha256_file(__file__),model_files=model_files,
                role='optional offline vocabulary discovery; not a deployment teacher dependency')
    out.write_text(json.dumps(result,indent=2)+'\n');print(a.scene,len(result['phrases']),out,flush=True)


if __name__=='__main__':main()
