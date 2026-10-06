"""Explicit cohort predictions before annotation scoring with a fixed point metric."""
import concurrent.futures,json,os,subprocess
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement
from radio_gs.data.scannet_source_region_semantics import load_scannet_raw_to_nyu40,official_vertex_nyu40_labels
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from prepare_independent_scannet_sources import ROOT,SCENES
PY='/root/miniconda3/envs/cybersim_agent/bin/python'

def main():
    jobs=[];config=ROOT/'configs';config.mkdir(exist_ok=True)
    for name in SCENES:
        maps=ROOT/'maps'/name;views=json.loads((ROOT/'evidence'/name/'views_0.json').read_text())['views']
        for mode in ['siglip_vote','adapter_vote','adapter_shared','adapter_shared_matched_mass','adapter_typed']:
            source=maps/('compiled/scene.json' if mode=='siglip_vote' else 'adapter/mask_adapter/scene.json')
            scene=json.loads(source.read_text());scene['native_observation_reliability']='entropy_information'
            scene.pop('observation_weighting',None);scene['source_streaming']=True
            scene['source_evidence']=dict(views=views,raster_hw=[96,128])
            if mode in ['adapter_shared','adapter_shared_matched_mass','adapter_typed']:
                scene['source_reliability_checkpoint']=str(ROOT/'scope_reliability.pt')
                scene['source_reliability_mode']=mode.removeprefix('adapter_')
                scene['map_contract']['supervised_source_reliability']=True
            dest=config/mode/f'{name}.scene.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(scene,indent=2)+'\n');jobs.append((mode,name,dest))
    evaluate_jobs(ROOT,SCENES,jobs)

def evaluate_jobs(ROOT,SCENES,jobs,evaluation_specs=None):
    """Predict every arm first, then consume annotations with the frozen scorer."""
    def predict(i):
        for mode,name,scene in jobs[i::6]:
            out=ROOT/'evaluation'/mode/name;out.mkdir(parents=True,exist_ok=True)
            with (out/'predict.log').open('w') as log:
                subprocess.run([PY,'-m','radio_gs.pipeline','--scene',str(scene),'--queries',str(ROOT/f'{name}.queries.json'),
                    '--access','source-assisted','--output',str(out/'query'),'--gpu',str(i)],env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(i)),stdout=log,stderr=subprocess.STDOUT,check=True)
            print(mode,name,'prediction complete',flush=True)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(predict,range(6)))
    # Open validation semantic annotations only after all predictions are saved.
    mapping=load_scannet_raw_to_nyu40(ROOT/'scannetv2-labels.combined.tsv')
    for name in SCENES:
        if evaluation_specs is not None:
            (ROOT/f'{name}.evaluation.json').write_text(Path(evaluation_specs[name]).read_text())
            continue
        ann=Path('/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small/annotations')/name
        mesh=PlyData.read(ann/f'{name}_vh_clean_2.ply');v=mesh['vertex'].data
        labels,report=official_vertex_nyu40_labels(scene_id=name,vertex_count=len(v),
            segmentation=json.loads((ann/f'{name}_vh_clean_2.0.010000.segs.json').read_text()),
            aggregation=json.loads((ann/f'{name}.aggregation.json').read_text()),raw_to_nyu40=mapping)
        values=np.empty(len(v),dtype=[('x','f4'),('y','f4'),('z','f4'),('label','u2')])
        for key in ['x','y','z']:values[key]=v[key]
        values['label']=labels;dest=ROOT/'evaluation'/f'{name}.labels.ply';PlyData([PlyElement.describe(values,'vertex')]).write(dest)
        (ROOT/f'{name}.evaluation.json').write_text(json.dumps(dict(scene=name,ground_truth=str(dest),splits=SPLITS,
            semantic_mapping=report,official_protocol_equivalence_verified=False,independent_mechanism_validation=True),indent=2)+'\n')
    def evaluate(i):
        for mode,name,_ in jobs[i::6]:
            out=ROOT/'evaluation'/mode/name
            with (out/'evaluate.log').open('w') as log:
                subprocess.run([PY,'scripts/research/evaluate_scannet_runtime.py','--readout',str(out/'query/category/typed_readout.pt'),
                    '--evaluation',str(ROOT/f'{name}.evaluation.json'),'--output',str(out/'evaluation'),'--point-only'],env=dict(os.environ,PYTHONPATH='.'),stdout=log,stderr=subprocess.STDOUT,check=True)
            print(mode,name,'evaluation complete',flush=True)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(evaluate,range(6)))
    rows={mode:{name:json.loads((ROOT/'evaluation'/mode/name/'evaluation/report.json').read_text())['point_metrics'] for name in SCENES} for mode in dict.fromkeys(j[0] for j in jobs)}
    macro={mode:{arm:{sp:sum(d[arm][sp]['miou'] for d in scenes.values())/len(scenes) for sp in ['19','15','10']} for arm in ['map','source']} for mode,scenes in rows.items()}
    (ROOT/'summary.json').write_text(json.dumps(dict(scenes=rows,macro=macro,completed=len(jobs),split=json.loads((ROOT/'split.json').read_text()),
        new_scene_for_this_mechanism=True,official_sota_equivalence=False,validation_parameter_selection=False),indent=2)+'\n');print(macro)

if __name__=='__main__':main()
