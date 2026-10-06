"""Native-query regional seeds corroborated by frozen three-view geometry.

A single correspondence-supported scope hypothesis per previously unseeded
query. This does not assert exhaustive multi-instance text retrieval.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
 key='lerf_'+a.scene;relation=load(Path('output/region_relations_next79')/a.scene/'relations.pt');manifest=json.loads(Path(relation['source_manifest']).read_text())
 data=load(Path('output/native_rebuild_next54/mixture_masscal')/key/'field.pt');root=Path(data['root']);q=F.normalize(load(root.parent.parent/'evaluation/text_online.encodings.pt')[0]['encoding'].float(),dim=-1)
 texts=load(Path('output/native_write_next75/all/calibrated')/key/'typed_readout.pt')['texts'];rows=[];nodes=[]
 for e in manifest['entries']:
  r=load(e['path']);rows.append(F.normalize(r['descriptors'].float(),dim=-1));nodes.extend((e['frame'],i,r['source']) for i in range(len(r['descriptors'])))
 scores=torch.cat(rows)@q.T;edges=torch.tensor(relation['equivalence_edges']);ij=edges[:,:2].long();out=Path('output/region_seeds_next80')/key;out.mkdir(parents=True,exist_ok=True);report=[]
 for qi,text in enumerate(texts):
  if load(Path('output/full_text_tracks_next79')/key/f'{qi}.pt')['seed_frame'] is not None:continue
  # Maximize the weakest native identity response in an already corroborated
  # pair; geometry is an admission condition, not a multiplier on cosine.
  pair=scores[ij,qi].amin(1);ei=int(pair.argmax());i,j=ij[ei].tolist();node=i if scores[i,qi]>=scores[j,qi] else j
  f,ri,source=nodes[node];r=load(source);sam=load(r['sam_path']);masks=unpack_masks(sam['packed_masks'],sam['mask_shape'][1]);mask=masks[int(r['native_proposal_indices'][ri])]
  confidence=float(pair[ei].clamp(0,1));record=dict(query=text,query_index=qi,seed_frame=f,seed_confidence=confidence,masks=torch.as_tensor(mask)[None].bool(),source='native regional pair; mutual geometric equivalence with third-view confirmation',pair_nodes=[i,j],pair_frames=[nodes[i][0],nodes[j][0]],native_cosines=scores[[i,j],qi].tolist(),geometry_weight=float(edges[ei,2]),single_scope_hypothesis=True)
  torch.save(record,out/f'{qi}.pt');report.append({k:v for k,v in record.items() if k!='masks'});print(text,f,confidence,flush=True)
 (out/'proposals.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
