"""Reconstruct every declared view on three new rooms; six GPU lanes."""
import concurrent.futures,hashlib,json,os,subprocess
from pathlib import Path
from prepare_independent_scannet_sources import ROOT,SCENES,cohort_arguments
PY='/root/miniconda3/envs/cybersim_agent/bin/python';SAM='/root/miniconda3/envs/sam3/bin/python'

def command(cmd,gpu,log):
    with Path(log).open('w') as file:
        subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=str(gpu)),stdout=file,stderr=subprocess.STDOUT,check=True)

def sam_lane(i):
    for name in SCENES:
        authority=ROOT/'sources'/name/'source_rgb.json';rows=json.loads(authority.read_text())['images'][i::6]
        if not rows:continue
        out=ROOT/'maps'/name;out.mkdir(parents=True,exist_ok=True)
        command([SAM,'-m','radio_gs.scripts.build_sam3_multiscale_hierarchy_cache','--source-authority',str(authority),
            '--source-authority-sha256',hashlib.sha256(authority.read_bytes()).hexdigest(),'--image-ids',','.join(r['image_id'] for r in rows),
            '--output-root',str(out/'source_sam'),'--manifest-name',f'shard_{i}.json','--checkpoint-path','checkpoints/sam3_modelscope/sam3.pt',
            '--checkpoint-sha256','9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e','--device','cuda:0'],i,out/f'sam_{i}.log')
        print(name,'SAM shard',i,'complete',flush=True)

def map_lane(i):
    name=SCENES[i];out=ROOT/'maps'/name
    command([PY,'scripts/research/build_scannet_native_map.py','--scene',name,'--source-root',str(ROOT/'sources'),'--output-root',str(ROOT/'maps')],i,out/'map.log')
    scene=json.loads((out/'scene.json').read_text());scene['native_category_prior']=dict(kind='mutually_exclusive_native_log_odds',temperature=100.)
    scene['native_observation_reliability']='entropy_information'
    scene['source_category_fusion']='visibility_weighted_vote';scene['source_identity_control']='detector_matched_mass';scene['source_streaming']=True;scene['source_query_batch_size']=4
    scene['source_evidence']=dict(raster_hw=[96,128],views=[])
    (out/'scene.json').write_text(json.dumps(scene,indent=2)+'\n')
    command([PY,'-m','radio_gs.pipeline.observation_field','--scene',str(out/'scene.json'),'--output',str(out/'compiled')],i,out/'compile.log')
    spec=json.loads((out/'compiled/scene.json').read_text())
    cameras=json.loads((ROOT/'sources'/name/'source_cameras.json').read_text())['records']
    spec['source_evidence']['views']=[dict(frame=c['frame_index'],source_rgb=c['source_image']) for c in cameras]
    (out/'compiled/scene.json').write_text(json.dumps(spec,indent=2)+'\n')
    command([PY,'-m','radio_gs.pipeline.adapter_observation_field','--scene',str(out/'compiled/scene.json'),'--regions',str(out/'native_regions'),'--output',str(out/'adapter')],i,out/'adapter.log')
    print(name,'maps complete',flush=True)

def detection_lane(i):
    name=SCENES[i];out=ROOT/'evidence'/name;out.mkdir(parents=True,exist_ok=True)
    from radio_gs.scannet_constants import NYU40_ID_TO_NAME,OPENGAUSSIAN_NYU40_CLASS_SPLITS as splits
    query=ROOT/f'{name}.queries.json';query.write_text(json.dumps(dict(queries=[dict(id=str(c),modality='text',intent='category',text=NYU40_ID_TO_NAME[c],class_id=c) for c in splits['19']]),indent=2)+'\n')
    scene=ROOT/'sources'/name/'provider_scene.json'
    command([SAM,'scripts/research/detect_consolidated_sources.py','--scene',str(scene),'--queries',str(query),'--checkpoint','checkpoints/sam3_modelscope/sam3.pt','--output',str(out/'detections'),'--height','96','--width','128'],i+3,out/'detection.log')
    command([PY,'scripts/research/build_category_source_evidence.py','--scene',str(scene),'--queries',str(query),'--output',str(out)],i+3,out/'identity.log')
    print(name,'source identity complete',flush=True)

if __name__=='__main__':
    args=cohort_arguments();ROOT=args.root;SCENES=args.scenes
    with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(sam_lane,range(6)))
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        jobs=[pool.submit(fn,i) for fn in [map_lane,detection_lane] for i in range(3)]
        for job in jobs:job.result()
    print('All independent maps and evidence complete',flush=True)
