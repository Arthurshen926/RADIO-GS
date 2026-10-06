"""Source-only review grid, never consumed by candidate selection."""
from PIL import Image,ImageDraw
from pathlib import Path
import json,argparse
p=argparse.ArgumentParser();p.add_argument("--seed-base",default="output/grounded_seeds_next85");p.add_argument("--output-base",default="output/next85");a=p.parse_args()
rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 root=Path(a.seed_base)/('lerf_'+scene)
 for r in json.loads((root/'proposals.json').read_text()):rows.append((r,root))
canvas=Image.new('RGB',(1000,260*((len(rows)+3)//4)),'white');draw=ImageDraw.Draw(canvas)
for i,(r,root) in enumerate(rows):
 x=i%4*250;y=i//4*260;draw.text((x+3,y+225),r['query'],fill='black');draw.text((x+3,y+241),'native fallback' if r['native_fallback'] else f"rank {r['selected_rank']}",fill='black')
 if not r['native_fallback']:
  crop=Image.open(root/f"{r['query_index']}_marked.jpg");crop.thumbnail((245,215));canvas.paste(crop,(x,y))
out=Path(a.output_base);out.mkdir(exist_ok=True);canvas.save(out/'source_choices.jpg')
