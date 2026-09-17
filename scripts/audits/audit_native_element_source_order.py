"""Source image-query ordering against explicit, conflict-filtered element labels.

This audits training evidence, not benchmark retrieval or geometric accuracy.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.training.correspondence_audit import training_query_conflicts
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--teachers',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 data=torch.load(a.root/'episodes.pt',weights_only=False);receipt=json.loads((a.root/'episodes.json').read_text());assert sha256_file(a.root/'episodes.pt')==receipt['sha256']
 info=json.loads((a.root/'teachers.json').read_text());assert sha256_file(a.root/'teachers.json')==data['teacher_manifest_sha256']
 observed=data['seen'][[info['frames'].index(f) for f in info['train_frames']]].any(0);rp=data['region_probability'].float();episodes=[e for e in data['episodes'] if e['split']=='train' and e['modality']=='image'];conflicts=training_query_conflicts(episodes,rp);result={}
 for name in a.teachers:
  path=a.root/name;teacher=torch.load(path,weights_only=False);assert teacher['episode_sha256']==receipt['sha256'];assert teacher['geometry_binding']==data['geometry_binding'];assert teacher['benchmark_data_used'] is False
  z=F.normalize(teacher['directions'].float(),dim=-1);q=F.normalize(F.linear(data['queries']['image'].float(),teacher['basis'].float()),dim=-1);rows=[]
  for e in episodes:
   qi=e['query_region'];pos=rp[e.get('positive_regions',[e['target_region']])].amax(0)>.7;neg=rp[e['negative_regions']].amax(0)>.7 if e['negative_regions'] else torch.zeros_like(pos);known=(pos^neg)&observed&teacher['valid'];key=('image',qi)
   if key in conflicts:known &= ~conflicts[key]
   pos &= known;neg &= known
   if not pos.any() or not neg.any():continue
   scores=z@q[qi];gap=float(scores[pos].mean()-scores[neg].mean());rows.append({'query_region':qi,'target_view':e['target_view'],'positive_elements':int(pos.sum()),'negative_elements':int(neg.sum()),'positive_mean_cosine':float(scores[pos].mean()),'negative_mean_cosine':float(scores[neg].mean()),'mean_cosine_gap':gap})
  result[name]={'sha256':sha256_file(path),'tasks_with_both_signs':len(rows),'nonpositive_mean_gap_tasks':sum(r['mean_cosine_gap']<=0 for r in rows),'mean_gap':sum(r['mean_cosine_gap'] for r in rows)/max(1,len(rows)),'observations':rows}
 a.output.write_text(json.dumps({'schema':'source_native_element_order.v1','benchmark_data_used':False,'episode_sha256':receipt['sha256'],'interpretation':'mean positive-negative gap is a source consistency diagnostic, not full separability or test accuracy','runs':result},indent=2)+'\n');print({k:{x:v for x,v in r.items() if x!='observations'} for k,r in result.items()})
if __name__=='__main__':main()
