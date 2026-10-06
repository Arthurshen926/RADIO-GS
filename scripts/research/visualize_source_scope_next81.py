"""Post-evaluation illustration; never consumed by prediction construction."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
cases=[('figurines','waldo'),('ramen','kamaboko'),('waldo_kitchen','pour-over vessel')];canvas=Image.new('RGB',(900,690),'white');draw=ImageDraw.Draw(canvas)
for row,(scene,query) in enumerate(cases):
 def objects(base):return json.loads((Path('output')/base/('lerf_'+scene)/'anchored/pipeline_calibrated/evaluation.json').read_text())['metrics']['hierarchical_pipeline']['selected']['objects']
 old=next(r for r in objects('region_seed_positive_next80') if r['query']==query);new=next(r for r in objects('source_scope_next81') if r['query']==query and r['frame']==old['frame']);gt=np.array(Image.open(old['gt_path']).convert('L'))>127
 for col,(title,record) in enumerate([('GT',None),('next80',old),('source initialization',new)]):
  if record is None:rgb=np.repeat((gt*255).astype('uint8')[...,None],3,2)
  else:
   pred=np.array(Image.open(record['pred_path']).convert('L').resize((gt.shape[1],gt.shape[0]),Image.Resampling.NEAREST))>10;rgb=np.zeros((*gt.shape,3),dtype='uint8');rgb[pred&gt]=(0,190,0);rgb[pred&~gt]=(220,0,0);rgb[~pred&gt]=(0,100,255)
  im=Image.fromarray(rgb);im.thumbnail((300,200));canvas.paste(im,(col*300,row*230));draw.text((col*300+3,row*230+201),query+' / '+title,fill='black')
out=Path('output/next81');out.mkdir(exist_ok=True);canvas.save(out/'selected_comparison.jpg')
