"""Source-only review sheet for the largest post-evaluation diagnostic gaps."""
import json,argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFilter


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    gaps=json.loads((root/'summary.json').read_text())['teacher_benchmark_gaps'][:6]
    sheet=Image.new('RGB',(1080,6*260),'white');draw=ImageDraw.Draw(sheet);records=[]
    for row,gap in enumerate(gaps):
        scene=gap['scene'];inputs=json.loads(Path(f'output/research_consolidation/source_run/{scene}/inputs.json').read_text())
        config=json.loads(Path(inputs['scene']).read_text());authority=json.loads(Path(config['source_rgb_authority']).read_text())
        paths={int(r['image_id'].split('_')[-1]):r['path'] for r in authority['images']}
        available=[]
        for i,frame in enumerate(config['source_frames']):
            if i%5:continue
            d=torch.load(Path(inputs['detections'])/f'identity_{frame}.pt',weights_only=False,map_location='cpu')
            r=d['instances'][d['queries'].index(gap['query'])]
            if len(r['scores']):available.append((frame,r))
        draw.text((4,row*260+4),f"{scene}: {gap['query']} | teacher {gap['teacher_iou']:.3f} / benchmark {gap['actual_iou']:.3f}",fill='black')
        for col,fraction in enumerate([.25,.5,.75]):
            frame,r=available[min(int(len(available)*fraction),len(available)-1)]
            im=Image.open(paths[frame]).convert('RGB');mask=r['masks'].float().amax(0).numpy()>.5
            binary=Image.fromarray((mask*255).astype('uint8')).resize(im.size,Image.Resampling.NEAREST)
            ys,xs=np.where(np.asarray(binary)>0)
            if not len(xs):continue
            box=(max(int(xs.min())-20,0),max(int(ys.min())-20,0),min(int(xs.max())+21,im.width),min(int(ys.max())+21,im.height))
            edge=np.asarray(binary.filter(ImageFilter.MaxFilter(3)))!=np.asarray(binary.filter(ImageFilter.MinFilter(3)))
            pixels=np.asarray(im).copy();pixels[edge]=[255,30,30];im=Image.fromarray(pixels).crop(box);im.thumbnail((352,210))
            sheet.paste(im,(col*360,row*260+26));draw.text((col*360+3,row*260+239),f"source frame {frame}, {len(r['scores'])} masks",fill='black')
            records.append(dict(scene=scene,query=gap['query'],frame=frame,source=paths[frame]))
    sheet.save(root/'teacher_gap_source_review.jpg')
    (root/'teacher_gap_source_review.json').write_text(json.dumps(dict(diagnostic_only=True,assistant_review_not_ground_truth=True,selection='six largest teacher/benchmark gaps; source views at 25/50/75 percentiles of valid held-out frames',records=records),indent=2)+'\n')


if __name__=='__main__':main()
