"""Keep high-resolution source evidence; save the matched 96x128 control.

Both rasters share detector predictions and low-resolution identity confidence.
Every authorized source view is processed. Each arm uses the same learned map.
"""
import concurrent.futures,json,os,subprocess
from pathlib import Path
PY='/root/miniconda3/envs/cybersim_agent/bin/python';SAM='/root/miniconda3/envs/sam3/bin/python'
ROOT=Path('output/paired_source_resolution');CONFIG=Path('configs/pipeline/paired_source_resolution')
SCENES=sorted(Path('configs/pipeline/scannet_identity_observations').glob('*.scene.json'))
GPUS=['0','1','2','3','4','5']
def worker(i):
    for path in SCENES[i::len(GPUS)]:
        name=path.name.removesuffix('.scene.json');out=ROOT/'evidence'/name;out.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=GPUS[i]);base=f'configs/consolidated/scannet_native/{name}.scene.json';query=f'configs/consolidated/scannet_native/{name}.queries.json'
        with (out/'build.log').open('w') as log:
            subprocess.run([SAM,'scripts/research/detect_consolidated_sources.py','--scene',base,'--queries',query,'--checkpoint','checkpoints/sam3_modelscope/sam3.pt','--output',str(out/'high/detections'),'--paired-low-output',str(out/'low/detections'),'--height','192','--width','256'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            for mode in ['low','high']:
                cmd=[PY,'scripts/research/build_category_source_evidence.py','--scene',base,'--queries',query,'--output',str(out/mode)]
                if mode=='high':cmd+=['--identity-from',str(out/'low')]
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for mode,shape in [('low',[96,128]),('high',[192,256])]:
            scene=json.loads((Path('output/adapter_observation_control')/name/'mask_adapter/scene.json').read_text())
            views=json.loads((out/mode/'views_0.json').read_text())['views'];scene['source_evidence']={**scene['source_evidence'],'views':views,'raster_hw':shape};scene['source_streaming']=True
            dest=CONFIG/mode/f'{name}.scene.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(scene,indent=2)+'\n')
        print(name,'paired source evidence complete',flush=True)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(worker,range(6)))
    subprocess.run([PY,'scripts/research/run_category_experiment.py','--configs',str(CONFIG),'--output',str(ROOT/'evaluation')],env=dict(os.environ,PYTHONPATH='.'),check=True)
