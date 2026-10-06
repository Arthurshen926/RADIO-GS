"""Train local object-boundary evidence on disjoint rooms; compile it offline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier

from radio_gs.querying.surface_consistency import surface_neighbours,edge_features,affinity_graph,EDGE_FEATURES

DATA=Path('output/learned_instance_additions')
ROOT=Path('output/surface_membership_study')


def training_inputs(scene):
    spec=json.loads((DATA/'design.json').read_text())
    row=PlyData.read(Path(spec['data_root'])/'scans'/f'{scene}.ply')['vertex'].data
    folder=DATA/'maps'/scene
    return (np.load(folder/'mesh_xyz.npy'),
        np.column_stack([row[k] for k in ['R','G','B']]).astype(np.float32)/255,
        np.load(folder/'element_appearance.npy',mmap_mode='r'),
        sparse.load_npz(folder/'visibility.npz').getnnz(axis=1)>0,
        np.asarray(row['label']))


def compile_training(scene):
    xyz,rgb,appearance,observed,labels=training_inputs(scene)
    surface=surface_neighbours(xyz);left,right=surface['left'],surface['right']
    valid=observed[left]&observed[right]&(labels[left]>=0)&(labels[right]>=0)
    truth=labels[left]==labels[right]
    seed=int.from_bytes(hashlib.sha256(('local-boundary/'+scene).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);chosen=[];weight=[]
    for label in [False,True]:
        ids=np.flatnonzero(valid&(truth==label));sample=rng.choice(ids,min(len(ids),10000),replace=False)
        chosen.extend(sample.tolist());weight.extend([.5/max(len(sample),1)]*len(sample))
    chosen=np.asarray(chosen,dtype=np.int64)
    x=edge_features(xyz,rgb,appearance,surface,chosen)
    folder=ROOT/'training';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{scene}.npz',x=x,y=truth[chosen],weight=weight)
    print('edge training',scene,len(chosen),int((~truth[chosen]).sum()),flush=True)


def fit():
    scenes=json.loads((DATA/'design.json').read_text())['fit']
    with ThreadPoolExecutor(3) as pool:list(pool.map(compile_training,scenes))
    rows=[np.load(ROOT/'training'/f'{s}.npz') for s in scenes]
    model=ExtraTreesClassifier(n_estimators=64,max_leaf_nodes=256,min_samples_leaf=32,
        max_features=1.,n_jobs=4,random_state=20261005)
    model.fit(np.concatenate([r['x'] for r in rows]),np.concatenate([r['y'] for r in rows]),
        sample_weight=np.concatenate([r['weight'] for r in rows]))
    joblib.dump(dict(estimator=model,features=EDGE_FEATURES,fit=scenes,
        supervision='same/different annotated whole instance; unknown excluded; equal room and edge-class mass',
        interpretation='positive pair conductance, not calibrated query membership',
        benchmark_labels_used=False),ROOT/'surface_affinity.joblib',compress=3)
    (ROOT/'fit.json').write_text(json.dumps(dict(rooms=len(scenes),pairs=sum(len(r['x']) for r in rows),
        per_room_training_edge_budget_per_class=10000,inference_edges_capped=False,
        feature_importance=dict(zip(EDGE_FEATURES,model.feature_importances_.tolist()))),indent=2)+'\n')


def benchmark_inputs(scene):
    from radio_gs.benchmarks.uqis.sensor import SensorIndex
    from radio_gs.pipeline.registered_observation_map import visible_vertices
    recipe=json.loads(Path('configs/pipeline/system_v8.json').read_text())
    job=next(j for j in recipe['jobs'] if j['benchmark']=='uqis' and j['name']==scene)
    source=Path(job['scene']).parent;spec=json.loads((source/'map.json').read_text())
    xyz=np.load(source/'mesh_xyz.npy');folder=ROOT/'benchmark_graphs'/scene;folder.mkdir(parents=True,exist_ok=True)
    package=Path(spec['configuration']['benchmark_package'])
    mapping=next(s for s in json.loads((package/'mapping/scenes.json').read_text())['scenes'] if s['scene_id']==scene)
    sensor=SensorIndex(mapping['sens_path']);width=spec['configuration']['source_rgb_width']
    total=np.zeros((len(xyz),3),np.float32);count=np.zeros(len(xyz),np.float32)
    for frame in spec['source_frames']:
        pose,depth,image=sensor.read(frame);height=round(image.height*width/image.width)
        ids,u,v=visible_vertices(xyz,pose,depth,sensor,(width,height),spec['configuration']['depth_tolerance_m'])
        colour=np.asarray(image.resize((width,height),Image.Resampling.LANCZOS),dtype=np.float32)/255
        total[ids]+=colour[v,u];count[ids]+=1
    rgb=total/np.maximum(count[:,None],1)
    np.save(folder/'source_rgb.npy',rgb)
    return xyz,rgb,np.load(DATA/'benchmark_local_appearance'/scene/'element_appearance.npy',mmap_mode='r'),count>0,None


def build_graph(scene,benchmark=False):
    start=time.monotonic()
    xyz,rgb,appearance,observed,labels=benchmark_inputs(scene) if benchmark else training_inputs(scene)
    surface=surface_neighbours(xyz);left,right=surface['left'],surface['right']
    valid=observed[left]&observed[right]
    # Every inference edge participates; chunks bound temporary feature memory.
    features=np.empty((len(left),len(EDGE_FEATURES)),np.float32)
    for start_index in range(0,len(left),32768):
        stop=min(start_index+32768,len(left))
        features[start_index:stop]=edge_features(xyz,rgb,appearance,surface,np.arange(start_index,stop))
    spatial=np.exp(-.5*surface['relative']**2)*valid
    rgb_scale=max(float(np.median(features[valid&(features[:,9]>0),9]**2)),1e-6)
    feature_scale=max(float(np.median(features[valid&(features[:,10]>0),10])),1e-6)
    fixed=spatial*np.exp(-features[:,9]**2/rgb_scale-features[:,10]/feature_scale)*(1-features[:,1])**2
    model=joblib.load(ROOT/'surface_affinity.joblib')['estimator'];model.n_jobs=1
    learned=spatial*np.maximum(2*model.predict_proba(features)[:,1]-1,0)
    folder=ROOT/('benchmark_graphs' if benchmark else 'graphs')/scene;folder.mkdir(parents=True,exist_ok=True)
    for name,value in [('fixed',fixed),('learned',learned)]:
        sparse.save_npz(folder/f'{name}.npz',affinity_graph(surface,value.astype(np.float32),len(xyz)))
    report=dict(scene=scene,elements=len(xyz),directed_candidate_edges=len(left),
        supported_edges=int(valid.sum()),source_views_added=0,geometry_changed=False,
        query_specific=False,graph_construction_uses_instance_labels=False,seconds=time.monotonic()-start)
    if not benchmark:
        known=valid&(labels[left]>=0)&(labels[right]>=0);same=labels[left]==labels[right]
        report['evaluation_only_edge_diagnosis']={name:dict(
            same_instance_mean_conductance=float(value[known&same].mean()),
            different_instance_mean_conductance=float(value[known&~same].mean()))
            for name,value in [('fixed',fixed),('learned',learned)]}
    (folder/'build.json').write_text(json.dumps(report,indent=2)+'\n');print('graphs',scene,round(report['seconds'],1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['fit','graphs'],required=True)
    p.add_argument('--benchmark',action='store_true');p.add_argument('--scenes',nargs='+');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if a.stage=='fit':fit();return
    design=json.loads((DATA/'design.json').read_text())
    scenes=a.scenes or ([j['name'] for j in json.loads(Path('configs/pipeline/system_v8.json').read_text())['jobs'] if j['benchmark']=='uqis']
        if a.benchmark else design['calibration']+design['validation'])
    with ThreadPoolExecutor(3) as pool:list(pool.map(lambda s:build_graph(s,a.benchmark),scenes))


if __name__=='__main__':main()
