"""Mutual, third-view-confirmed regional equivalence; containment stays separate."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4);start=time.monotonic()
 manifest_path=Path('output/region_bank_next78')/a.scene/'manifest.json';manifest=json.loads(manifest_path.read_text());n=manifest['gaussians'];views={};descriptors={};offsets={};offset=0
 for e in manifest['entries']:
  r=torch.load(e['path'],map_location='cpu',weights_only=False);ids=r['ids'].numpy();rr=r['rows'].numpy();cc=r['columns'].numpy();m=sp.csr_matrix((r['values'].numpy(),(ids[rr],cc)),shape=(n,len(r['descriptors'])));mass=np.zeros(n,np.float32);mass[ids]=r['mass'].numpy();views[e['frame']]=(m,mass)
  descriptors[e['frame']]=F.normalize(r['descriptors'].float(),dim=1);offsets[e['frame']]=offset;offset+=len(r['descriptors'])
 reference=json.loads((Path('output/region_correspondence_next77')/a.scene/'correspondence.json').read_text())
 def compare(fa,fb):
  ma,wa=views[fa];mb,wb=views[fb];common=np.sqrt(wa*wb);am=np.asarray(ma.T@common).ravel();bm=np.asarray(mb.T@common).ravel();intersection=(ma.T@mb.multiply(common[:,None])).toarray();ca=intersection/np.maximum(am[:,None],1e-8);cb=intersection/np.maximum(bm[None,:],1e-8);dice=2*intersection/np.maximum(am[:,None]+bm[None,:],1e-8)
  return dice,ca,cb
 equivalent=[];containment=[];stats=[]
 for row in reference['records']:
  fa,fb,fc=row['frame'],row['association_view'],row['confirmation_view'];ab,coverage_a,coverage_b=compare(fa,fb);ac,_,_=compare(fa,fc);bc,_,_=compare(fb,fc)
  match=ab.argmax(1);back=ab.argmax(0);ai=np.arange(len(match));pair=ab[ai,match];third=np.minimum(ac,bc[match]).max(1);ca=coverage_a[ai,match];cb=coverage_b[ai,match];mutual=back[match]==ai
  # Fixed source-only relation criteria, shared by every scene and query.
  equal=mutual&(ca>=.7)&(cb>=.7)&(third>=.7)
  semantic=(descriptors[fa]*descriptors[fb][torch.from_numpy(match)]).sum(1).numpy().clip(0,1)
  for i in np.where(equal)[0]:equivalent.append((offsets[fa]+int(i),offsets[fb]+int(match[i]),float(pair[i]*third[i]*semantic[i]**2)))
  # Directional geometric containment is recorded, never an identity edge.
  aa,bb=np.where((coverage_a>=.8)&(coverage_b<.5))
  containment.extend((offsets[fa]+int(i),offsets[fb]+int(j),float(coverage_a[i,j]),float(coverage_b[i,j])) for i,j in zip(aa,bb))
  aa,bb=np.where((coverage_b>=.8)&(coverage_a<.5))
  containment.extend((offsets[fb]+int(j),offsets[fa]+int(i),float(coverage_b[i,j]),float(coverage_a[i,j])) for i,j in zip(aa,bb))
  stats.append(dict(frame=fa,association_view=fb,confirmation_view=fc,regions=len(match),mutual=int(mutual.sum()),equivalent=int(equal.sum())))
  print(a.scene,fa,'equivalent',int(equal.sum()),flush=True)
 # Deduplicate undirected equivalence edges; reciprocal camera choices are not
 # independent repeated votes. Containment remains a separate directed relation.
 edges={}
 for i,j,w in equivalent:
  key=(min(i,j),max(i,j));edges[key]=max(edges.get(key,0.),w)
 original=torch.cat([descriptors[e['frame']] for e in manifest['entries']]);sums=torch.zeros_like(original);mass=torch.zeros(len(original),1)
 if edges:
  ij=torch.tensor(list(edges),dtype=torch.long);weight=torch.tensor(list(edges.values()))[:,None]
  sums.index_add_(0,ij[:,0],original[ij[:,1]]*weight);sums.index_add_(0,ij[:,1],original[ij[:,0]]*weight)
  mass.index_add_(0,ij[:,0],weight);mass.index_add_(0,ij[:,1],weight)
 propagated=F.normalize(original+sums/mass.clamp_min(1e-8),dim=1)
 replacements={f:propagated[offsets[f]:offsets[f]+len(d)] for f,d in descriptors.items()}
 out=Path('output/region_relations_next79')/a.scene;out.mkdir(parents=True,exist_ok=True);torch.save(replacements,out/'descriptors.pt')
 torch.save(dict(equivalence_edges=[(i,j,w) for (i,j),w in edges.items()],containment_edges=containment,offsets=offsets,source_manifest=str(manifest_path)),out/'relations.pt')
 (out/'report.json').write_text(json.dumps(dict(source_manifest=str(manifest_path),source_frames=manifest['source_frames'],regions=len(original),equivalence_edges=len(edges),regions_with_neighbors=int((mass[:,0]>0).sum()),containment_edges=len(containment),third_view_excluded_from_pair_assignment=True,thresholds=dict(mutual_coverage=.7,confirmation=.7,containment_in=.8,containment_out=.5),propagation='one original-plus-normalized-neighbor native-vector average; no containment propagation',query_independent=True,verified_instance_identity=False,seconds=time.monotonic()-start,views=stats),indent=2)+'\n')
if __name__=='__main__':main()
