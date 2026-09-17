"""Reuse RGB/text teacher predictions only when all their true inputs match.

Region-indexed labels are never reused: they must be compiled again afterward.
"""
import argparse,json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 path=a.parent/'source_grounded_text.pt';receipt=json.loads(path.with_suffix('.json').read_text());assert sha256_file(path)==receipt['sha256'];g=torch.load(path,weights_only=False)
 old=json.loads((a.parent/'teachers.json').read_text());new=json.loads((a.root/'teachers.json').read_text())
 assert g['teacher_manifest_sha256']==sha256_file(a.parent/'teachers.json')
 assert g['source_only'] and not g['benchmark_labels_used']
 assert new['train_frames']==old['train_frames']==g['frames']
 assert sha256_file(a.parent/'responsibility.json')==sha256_file(a.root/'responsibility.json')==new['authority_sha256']
 assert sha256_file(a.parent/'source_frames.json')==sha256_file(a.root/'source_frames.json')
 assert sha256_file(a.parent/'native_query_encoder.json')==sha256_file(a.root/'native_query_encoder.json')
 r=new['teacher_files']['siglip'];assert sha256_file(a.root/r['path'])==r['sha256'];sig=torch.load(a.root/r['path'],weights_only=False);assert sig['queries']==g['queries']
 source=json.loads((a.root/'source_frames.json').read_text());frames={x['frame_idx']:x for x in source['frames']}
 for rec in g['source_records']:
  frame=frames[rec['frame']];assert frame['source_sha256']==rec['sha256']==sha256_file(Path(source['image_dir'])/frame['source_file'])
 assert g['grounding_phrases']==[q.removeprefix('a photo of a ') for q in g['queries']]
 auth=json.loads((a.root/'responsibility.json').read_text());assert g['geometry_binding']==auth['metadata']['frozen_geometry_binding']
 assert tuple(g['probability'].shape[-2:])==(auth['metadata']['feature_height'],auth['metadata']['feature_width'])
 out=a.root/'source_grounded_text.pt'
 if out.exists():raise FileExistsError(out)
 result=dict(g,teacher_manifest_sha256=sha256_file(a.root/'teachers.json'),inventory_rebinding={'source_prediction_path':str(path),'source_prediction_sha256':receipt['sha256'],'reason':'only region inventory changed; identical RGB, phrases, source frames, encoder and projection authority','implementation_sha256':sha256_file(__file__)})
 torch.save(result,out);out.with_suffix('.json').write_text(json.dumps({'schema':g['schema'],'sha256':sha256_file(out),'teacher_manifest_sha256':result['teacher_manifest_sha256'],'source_prediction_sha256':receipt['sha256'],'teacher_probabilities_unchanged':torch.equal(g['probability'],result['probability'])},indent=2)+'\n')
 print(out,flush=True)
if __name__=='__main__':main()
