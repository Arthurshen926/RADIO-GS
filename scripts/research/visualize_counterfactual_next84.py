"""Source-only matched-window illustrations, not predictor inputs."""
from pathlib import Path
from PIL import Image,ImageDraw
cases=[('ramen',11,'sake cup'),('waldo_kitchen',4,'ketchup'),('waldo_kitchen',10,'pour-over vessel')];im=Image.new('RGB',(900,780),'white');d=ImageDraw.Draw(im)
for row,(scene,qi,label) in enumerate(cases):
 for col,(arm,suffix,title) in enumerate([('raw','chosen','Raw likelihood'),('contrast','chosen','Erasure gain'),('contrast','erased','Object erased')]):
  crop=Image.open(Path('output/counterfactual_seeds_next84')/arm/('lerf_'+scene)/f'{qi}_{suffix}.jpg');crop.thumbnail((290,220));im.paste(crop,(col*300,row*260));d.text((col*300+3,row*260+225),label+' / '+title,fill='black')
out=Path('output/next84');out.mkdir(exist_ok=True);im.save(out/'source_erasure.jpg')
