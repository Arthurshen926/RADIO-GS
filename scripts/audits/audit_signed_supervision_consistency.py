"""Audit conflicts between authorized signed pixels and lifted region labels."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(4);root=a.root;receipt=json.loads((root/'episodes.json').read_text());assert sha256_file(root/'episodes.pt')==receipt['sha256'];d=torch.load(root/'episodes.pt',weights_only=False)
 assert sha256_file(root/'teachers.json')==d['teacher_manifest_sha256'];info=json.loads((root/'teachers.json').read_text());assert sha256_file(root/'responsibility.json')==info['authority_sha256'];auth=json.loads((root/'responsibility.json').read_text())
 trainviews=[info['frames'].index(f) for f in info['train_frames']];observed=d['seen'][trainviews].any(0);hits=[]
 for r in auth['views']:
  path=root/r['relative_path'];assert sha256_file(path)==r['sha256'];hits.append(torch.load(path,weights_only=False))
 rows=[]
 for i,e in enumerate(d['episodes']):
  if e['modality']!='prompt':continue
  q=e['query_region'];v=int(d['proposal_views'][q]);hit=hits[v];pr=d['region_probability'][e['target_region']].float();nr=d['region_probability'][e['negative_regions']].float().amax(0) if e['negative_regions'] else torch.zeros_like(pr)
  known=(pr>.7)^(nr>.7);target=pr>.7;seeds=[]
  for sign,pixel in zip([1,0],d['queries']['seed_pixels'][q]):
   selected=hit['pixel_ids']==int(pixel);ids=hit['gaussian_ids'][selected].long();weight=hit['base_weights'][selected].float();total=weight.sum();keep=observed[ids];weight=weight*keep;mass=weight.sum()
   conflict=known[ids]&(target[ids]!=bool(sign));aligned=known[ids]&(target[ids]==bool(sign))
   seeds.append({'sign':sign,'observed_mass_fraction':float(mass/total.clamp_min(1e-8)),'conflicting_target_label_fraction':float(weight[conflict].sum()/mass.clamp_min(1e-8)),'aligned_target_label_fraction':float(weight[aligned].sum()/mass.clamp_min(1e-8))})
  rows.append({'episode':i,'split':e['split'],'query_region':q,'target_region':e['target_region'],'seeds':seeds})
 result={'episode_sha256':receipt['sha256'],'benchmark_ground_truth':False,'episodes':rows,'summary':{}}
 for split in ['train','validation']:
  subset=[r for r in rows if r['split']==split];result['summary'][split]={'episodes':len(subset),'episodes_with_majority_seed_label_conflict':sum(any(s['observed_mass_fraction']>.5 and s['conflicting_target_label_fraction']>.5 for s in r['seeds']) for r in subset),'episodes_with_any_seed_label_conflict_over_10pct':sum(any(s['observed_mass_fraction']>.5 and s['conflicting_target_label_fraction']>.1 for s in r['seeds']) for r in subset)}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['summary'])
if __name__=='__main__':main()
