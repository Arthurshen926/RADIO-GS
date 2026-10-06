"""Post-prediction region identity and geometric-link diagnosis.

Annotations are used only to measure frozen proposals/links. They do not change
predictions, correspondence admission, confidence or model parameters.
"""
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import torch
from torch.nn import functional as F
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.native_query_encoders import encode_category_text


@torch.inference_mode()
def measure(root,name):
    scene=json.loads((root/'configs/v6'/f'{name}.scene.json').read_text())
    spec=json.loads((root/f'{name}.evaluation.json').read_text())
    vertices=PlyData.read(spec['ground_truth'])['vertex'].data
    labels=np.asarray(vertices['label']).astype(np.int64)
    xyz=np.stack([vertices[k] for k in ['x','y','z']],1)
    geometry=FrozenGaussianGeometry.load(scene['geometry']['path'],expected_sha256=scene['geometry']['sha256'],device='cpu')
    _,binding=cKDTree(geometry.xyz.numpy()).query(xyz,workers=4)
    ann=Path('/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small/annotations')/name
    segments=np.asarray(json.loads((ann/f'{name}_vh_clean_2.0.010000.segs.json').read_text())['segIndices'])
    objects=json.loads((ann/f'{name}.aggregation.json').read_text())['segGroups']
    instance=np.zeros(len(segments),dtype=np.int64)
    for i,obj in enumerate(objects):instance[np.isin(segments,obj['segments'])]=i+1
    class_count=int(labels.max())+1;instance_count=len(objects)+1;n=len(geometry.xyz)
    element_class=np.bincount(binding*class_count+labels,minlength=n*class_count).reshape(n,class_count).astype(np.float32)
    element_instance=np.bincount(binding*instance_count+instance,minlength=n*instance_count).reshape(n,instance_count).astype(np.float32)
    auxiliary=json.loads(Path(scene['local_auxiliary_scene']).read_text())
    menu=json.loads((root/f'{name}.queries.json').read_text())['queries']
    texts=[q['text'] for q in menu];class_ids=np.asarray([q['class_id'] for q in menu])
    query=encode_category_text(auxiliary,texts,device='cuda')
    folder=root/'region_observations'/name;graph=torch.load(folder/'graph.pt',map_location='cpu')
    counts=[];instances=[];emitting=[];native=[]
    for frame in graph['source_frames']:
        row=torch.load(folder/f'frame_{frame:05d}.pt',map_location='cpu');group=row['instance'];nr=len(group['masks'])
        rows=row['element_rows'].numpy();cols=row['region_columns'].numpy()
        membership=csr_matrix((np.ones(len(rows),np.float32),(cols,rows)),shape=(nr,n))
        counts.append(membership@element_class);instances.append(membership@element_instance)
        emitting.append(group['emitting'][:,[row['queries'].index(t) for t in texts]])
        native.append((100*F.normalize(group['descriptors'].float().cuda(),dim=-1)@query.T).softmax(1).cpu())
    counts=np.concatenate(counts);instances=np.concatenate(instances)
    emitting=torch.cat(emitting);native=torch.cat(native)
    edges=graph['edges'];weight=graph['weights'];linked=native.clone();mass=torch.ones(len(native),1)
    for a,b in [(edges[:,0],edges[:,1]),(edges[:,1],edges[:,0])]:
        linked.index_add_(0,a,weight[:,None]*native[b]);mass.index_add_(0,a,weight[:,None])
    linked/=mass
    total=counts.sum(1);major=counts.argmax(1);purity=counts.max(1)/np.maximum(total,1)
    obj=instances.argmax(1);objpurity=instances.max(1)/np.maximum(instances.sum(1),1)
    predictions={key:class_ids[score.argmax(1).numpy()] for key,score in dict(emitter=emitting,native=native,verified=emitting*native,linked_verified=emitting*linked,linked_native=linked).items()}
    result=dict(nodes=len(counts),empty_projected_nodes=int((total==0).sum()),edges=len(edges),source_views=len(graph['source_frames']))
    for key,valid in dict(dominant_class_in_menu=(total>0)&np.isin(major,class_ids),pure_class_in_menu=(total>0)&np.isin(major,class_ids)&(purity>=.8)).items():
        result[key]=dict(count=int(valid.sum()),accuracy={k:float((p[valid]==major[valid]).mean()) for k,p in predictions.items()})
    ei,ej=edges.numpy().T
    clean=(obj[ei]>0)&(obj[ej]>0)&(objpurity[ei]>=.8)&(objpurity[ej]>=.8)
    pureclass=(major[ei]>0)&(major[ej]>0)&(purity[ei]>=.8)&(purity[ej]>=.8)
    result['correspondence']=dict(pure_instance_endpoint_edges=int(clean.sum()),same_instance_fraction=float((obj[ei[clean]]==obj[ej[clean]]).mean()),
        pure_class_endpoint_edges=int(pureclass.sum()),same_class_fraction=float((major[ei[pureclass]]==major[ej[pureclass]]).mean()))
    known=(total>0)&np.isin(major,class_ids)&(purity>=.8)
    before=predictions['verified']==major;after=predictions['linked_verified']==major
    result['linked_identity_change_on_pure_nodes']=dict(correct_to_wrong=int((known&before&~after).sum()),wrong_to_correct=int((known&~before&after).sum()))
    result['identity_transport']=dict(single_emitting_label_nodes=int(((emitting>0).sum(1)==1).sum()),native_top1_changed_by_link=int((native.argmax(1)!=linked.argmax(1)).sum()),verified_top1_changed_by_link=int(((emitting*native).argmax(1)!=(emitting*linked).argmax(1)).sum()),edge_native_top1_disagreement=float((native[edges[:,0]].argmax(1)!=native[edges[:,1]].argmax(1)).float().mean()),mean_edge_native_total_variation=float((native[edges[:,0]]-native[edges[:,1]]).abs().sum(1).mean()/2))
    result['scope_purity']=dict(class_pure_fraction=float((purity[total>0]>=.8).mean()),instance_pure_fraction=float((objpurity[total>0]>=.8).mean()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('output/region_identity_transport'));a=p.parse_args();torch.set_num_threads(4)
    summary=json.loads((a.root/'summary.json').read_text())
    rows={}
    for name in summary['scenes']['v6']:
        rows[name]=measure(a.root,name);print(name,rows[name],flush=True)
    (a.root/'region_identity_outcomes.json').write_text(json.dumps(dict(scenes=rows,post_prediction_only=True,
        ground_truth_used_only_for_measurement=True,geometry_binding='same GT-point to nearest element as frozen evaluator',
        region_truth='majority GT points assigned to projected member elements; not independent pixel annotation',
        purity_threshold=.8,void_in_purity_denominator=True,frame_links='adjacent views; NOT held-out-view validation'),indent=2)+'\n')

if __name__=='__main__':main()
