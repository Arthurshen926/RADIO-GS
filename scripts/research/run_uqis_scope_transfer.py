"""One bounded scope-completion candidate, same map and no new source inputs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

PY='/root/miniconda3/envs/cybersim_agent/bin/python'
ROOT=Path('output/uqis_scope_transfer_v1')
BASE=Path('output/uqis_map_only_v1')
PUBLIC=Path('output/scannet_uqis_9_v0_3_candidate/release_final/method')


def main():
    global ROOT,BASE,PUBLIC
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--maps-root',type=Path,default=BASE)
    p.add_argument('--package',type=Path,default=PUBLIC.parent)
    p.add_argument('--gpus',nargs='+',type=int,default=[2,3,4,5])
    a=p.parse_args();ROOT=a.output;BASE=a.maps_root;PUBLIC=a.package/'method'
    ROOT.mkdir(parents=True,exist_ok=True)
    maps=sorted((BASE/'maps').glob('*/map.json'))
    names=['native_semantic','anchor_comembership','anchor_correspondence']
    def lane(i):
        gpu=str(a.gpus[i])
        for map_file in maps[i::len(a.gpus)]:
            scene=map_file.parent.name;out=ROOT/'scenes'/scene;out.mkdir(parents=True,exist_ok=True)
            with (out/'query.log').open('w') as log:
                subprocess.run([PY,'-u','-m','radio_gs.pipeline','--scene',str(map_file),'--queries',str(PUBLIC),
                    '--access','map-only','--scope-transfer','--gpu',gpu,'--output',str(out)],
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
            for name in names:
                target=ROOT/name/'predictions';target.mkdir(parents=True,exist_ok=True)
                for file in (out/'registered'/name/'predictions').glob('*.npy'):
                    shutil.copyfile(file,target/file.name)
            print(scene,'three readouts complete via public runtime',flush=True)
    with ThreadPoolExecutor(len(a.gpus)) as pool:list(pool.map(lane,range(len(a.gpus))))
    (ROOT/'predictions_complete.json').write_text(json.dumps(dict(scenes=len(maps),
        all_predictions_before_this_scoring=True,development_followup_after_v1_diagnosis=True,
        benchmark_package=str(a.package),map_root=str(BASE),
        new_source_images=0,new_teachers=0,parameter_sweeps=0),indent=2)+'\n')
    for name in names:
        with (ROOT/f'{name}.evaluation.log').open('w') as log:
            subprocess.run([PY,'-m','radio_gs.benchmarks.uqis.evaluate','--package',str(PUBLIC.parent),
                '--predictions',str(ROOT/name/'predictions'),'--output',str(ROOT/name/'result.json')],
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
