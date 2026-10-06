"""Official VALA geometry/source-budget exchange; no upstream method edits."""
import argparse,json,os,runpy,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/root/RADIO-GS');UP=Path('/root/baselines/VALA');OUT=ROOT/'output/vala_exchange_next48'
DATA=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs')
PYTHON='/root/miniconda3/envs/cybersim_agent/bin/python'
SCENES=['figurines','ramen','teatime','waldo_kitchen']

def prepare(scene,arm):
    import torch
    from torch import nn
    from arguments import ModelParams,OptimizationParams
    from scene.gaussian_model import GaussianModel
    model=OUT/arm/scene;model.mkdir(parents=True,exist_ok=True)
    source=OUT/'source'/scene;sparse=source/'sparse/0';sparse.mkdir(parents=True,exist_ok=True)
    for name in ['images','langsplat','language_features']:
        link=source/name
        if not link.exists():link.symlink_to(DATA/scene/name,target_is_directory=True)
    for src in (DATA/scene/'sparse/0').iterdir():
        link=sparse/src.name
        if src.name!='test.txt' and not link.exists():link.symlink_to(src)
    tests=[p.stem for p in sorted((DATA/'label'/scene).glob('frame_*.json'))]
    (sparse/'test.txt').write_text('\n'.join(tests)+'\n')
    field=torch.load(ROOT/'output/taskwise_scope_composition_next38'/('lerf_'+scene)/'field.pt',map_location='cpu',weights_only=False)
    teachers=json.loads((ROOT/field['root']/'teachers.json').read_text())
    frames=[f'frame_{f:05d}' for f in teachers['train_frames']]
    (model/'source_frames.json').write_text(json.dumps(frames))
    parser=argparse.ArgumentParser();mp=ModelParams(parser);op=OptimizationParams(parser);args=parser.parse_args([])
    args.source_path=str(source);args.model_path=str(model);args.data_device='cpu';args.eval=True
    (model/'cfg_args').write_text(str(argparse.Namespace(**vars(mp.extract(args)))))
    geo=GaussianModel(3)
    if arm in ['moge20','moge_fit20']:
        report=json.loads((ROOT/field['geometry_report']).read_text())['results']['independent_scale']
        geometry=ROOT/'output/carrier_fit_next48'/scene/'geometry.pt' if arm=='moge_fit20' else ROOT/report['geometry']
        state=torch.load(geometry,map_location='cuda',weights_only=False)['state_dict'];n=len(state['xyz'])
        geo._xyz=nn.Parameter(state['xyz']);geo._rotation=nn.Parameter(state['quaternions'])
        geo._scaling=nn.Parameter(state['scales'].log());geo._opacity=nn.Parameter(torch.logit(state['opacities']))
        geo._features_dc=nn.Parameter(torch.zeros(n,1,3,device='cuda'))
        if arm=='moge_fit20':
            from utils.sh_utils import RGB2SH
            geo._features_dc=nn.Parameter(RGB2SH(torch.load(ROOT/'output/carrier_fit_next48'/scene/'rgb.pt',map_location='cuda'))[:,None])
        geo._features_rest=nn.Parameter(torch.zeros(n,15,3,device='cuda'));geo.active_sh_degree=3
        geo.save_ply(str(model/'point_cloud/iteration_30000/point_cloud.ply'))
    else:
        geometry=ROOT/'output/3dgs_models'/scene/'point_cloud/iteration_30000/point_cloud.ply'
        link=model/'point_cloud'
        if not link.exists():link.symlink_to(geometry.parents[1],target_is_directory=True)
        geo.load_ply(str(geometry))
    geo.training_setup(op.extract(args));geo.max_radii2D=torch.zeros(len(geo.get_xyz),device='cuda')
    torch.save((geo.capture_rgb(),30000),model/'chkpnt30000.pth')
    meta=dict(scene=scene,arm=arm,geometry=str(geometry),source_frames=frames if arm!='rgb_all' else 'all non-annotation frames',test_frames=tests,
              geometry_training_split_reestablished=False,official_author_reproduction=False,upstream_method_unchanged=True,
              rgb_sh_placeholder=arm=='moge20',purpose='source-budget and frozen-geometry exchange with fixed VALA semantics/readout')
    (model/'experiment.json').write_text(json.dumps(meta,indent=2)+'\n')

def invoke(scene,arm,stage,level):
    # Filter the official scene callback before image loading; retain evaluation
    # cameras. Neither features nor predictions use annotation mask contents.
    import scene as scene_module
    model=OUT/arm/scene;source=OUT/'source'/scene
    if arm!='rgb_all':
        names=set(json.loads((model/'source_frames.json').read_text()))
        original=scene_module.sceneLoadTypeCallbacks['Colmap']
        def subset(*args,**kwargs):
            info=original(*args,**kwargs)
            train=[c for c in info.train_cameras if c.image_name in names]
            print('Fixed source subset:',[c.image_name for c in train],flush=True)
            return info._replace(train_cameras=train)
        scene_module.sceneLoadTypeCallbacks['Colmap']=subset
    common=['-s',str(source),'-m',str(model),'--iteration','30000','--eval']
    if stage=='extract':
        sys.argv=['gaussian_feature_extractor.py']+common+['--feature_level',str(level),'--use_efficient']
        runpy.run_path(str(UP/'gaussian_feature_extractor.py'),run_name='__main__')
    else:
        sys.argv=['render']+common+['--mask_thresh','0.6','--scene_name',scene,'--dataset_name','lerf_ovs','--base_dir',str(model),'--output_dir',str(model),'--skip_train']
        runpy.run_module('eval.render_lerf_by_text_langsplat',run_name='__main__')

def worker(job):
    gpu,scene,arm=job;model=OUT/arm/scene;model.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MAX_JOBS='4',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONPATH=f'{UP}/submodules/gsplat:{UP}:{ROOT}')
    script=str(Path(__file__).resolve());base=[PYTHON,script,'--scene',scene,'--arm',arm]
    commands=[base+['--stage','prepare']]+[base+['--stage','extract','--level',str(i)] for i in (1,2,3)]+[base+['--stage','render'],
        [PYTHON,'-m','eval.compute_lerf_iou','--gt_dir',str(OUT/arm),'--pred_dir',str(OUT/arm),'--output_dir',str(OUT/arm),'--iteration','30000','--mask_thresh','0.6','--ablation_type','none','--json_dir',str(DATA/'label'),'--scene_name',scene]]
    rows=[]
    for j,c in enumerate(commands):
        start=time.monotonic()
        with (model/f'step{j}.log').open('w') as log:rc=subprocess.run(c,cwd=UP,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        rows.append(dict(scene=scene,arm=arm,command=c,exit_code=rc,seconds=time.monotonic()-start));print(scene,arm,j,rc,flush=True)
        if rc:break
    (model/'execution.json').write_text(json.dumps(rows,indent=2)+'\n');return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--arm');p.add_argument('--stage');p.add_argument('--level',type=int);p.add_argument('--gpus',default='0,1,2,3,4,5');a=p.parse_args()
    if a.stage=='prepare':prepare(a.scene,a.arm)
    elif a.stage:invoke(a.scene,a.arm,a.stage,a.level)
    else:
        jobs=[(s,arm) for arm in ([a.arm] if a.arm else ['rgb20','moge20','rgb_all']) for s in ([a.scene] if a.scene else SCENES)];gpus=a.gpus.split(',');rows=[]
        for start in range(0,len(jobs),len(gpus)):
            with ThreadPoolExecutor(len(gpus)) as pool:rows+=sum(pool.map(worker,[(gpus[i],s,arm) for i,(s,arm) in enumerate(jobs[start:start+len(gpus)])]),[])
        (OUT/('execution_'+a.arm+'_'+(a.scene or 'all')+'.json' if a.arm else 'execution.json')).write_text(json.dumps(rows,indent=2)+'\n')
