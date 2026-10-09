"""Fresh copied-export replay of real native inputs and surface registration."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image
import torch

from radio_gs.query_field.adapters import NativeQueryEncoder,point_3d
from radio_gs.query_field.contracts import QueryPacket
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.query_field.surface_point import SurfacePointAdapter
from scripts.research.run_query_field_native_modes import ROOT,read,write,receipt,design


@torch.no_grad()
def capture(directory,fixture):
    runtime=QueryFieldRuntime.load(directory,'cuda');adapter=SurfacePointAdapter(runtime);result={}
    for modality in ['text','image','point_2d','point_3d']:
        q=fixture[modality]
        if modality in ['text','image']:packet=QueryPacket(modality,**q)
        elif modality=='point_2d':packet,_=adapter.packet(q['positive_pixel_uv'],q['camera_intrinsics'],q['camera_to_world'])
        else:packet=point_3d(runtime,q['point_world_xyz'])
        prediction=runtime.query(packet)
        result[modality]=dict(scores=prediction.membership_scores.cpu(),selected=prediction.selected.cpu(),
            candidates=prediction.candidate_indices.cpu(),weights=prediction.candidate_weights.cpu())
    return result


def child(args):
    forbidden=[Path(p).resolve() for p in args.deny];attempts=[]
    def guard(event,values):
        if event!='open' or not isinstance(values[0],(str,bytes)):return
        path=Path(os.fsdecode(values[0])).resolve()
        if any(path==p or p in path.parents for p in forbidden):
            attempts.append(str(path));raise PermissionError('source path denied')
    sys.addaudithook(guard)
    fixture=torch.load(args.fixture,map_location='cpu',weights_only=True)
    torch.save(capture(args.field,fixture),args.output)
    print(json.dumps(dict(forbidden_open_attempts=attempts)),flush=True)


def main():
    read(ROOT/'scoring_complete.json');d=design();public=Path(d['package'])/'method';scene=d['fields'][0]['scene']
    out=ROOT/'validation/runtime';out.mkdir(parents=True,exist_ok=False)
    queries={m:next(q for q in read(public/'queries'/(m+'.json'))['queries'] if q['scene_id']==scene)
             for m in ['text','image','point_2d','point_3d']}
    encoder=NativeQueryEncoder(d['encoder'],'cuda');fixture={}
    packet=encoder.text(queries['text']['expression']);fixture['text']=dict(semantic=packet.semantic.cpu())
    q=queries['image'];image_path=public/q['crop_rgb_path']
    if receipt(image_path)['sha256']!=q['crop_rgb_sha256']:raise ValueError('current image changed')
    packet=encoder.image(Image.open(image_path).convert('RGB'),native_resolution=True)
    fixture['image']=dict(semantic=packet.semantic.cpu(),local=packet.local.cpu(),local_weights=packet.local_weights.cpu())
    fixture.update({m:queries[m] for m in ['point_2d','point_3d']})
    del encoder,packet;gc.collect();torch.cuda.empty_cache()
    torch.save(fixture,out/'fixture.pt');directory=ROOT/'fits'/scene/'measure/field'
    expected=capture(directory,fixture);torch.save(expected,out/'original.pt');gc.collect();torch.cuda.empty_cache()
    repo=Path.cwd().resolve()
    with tempfile.TemporaryDirectory(prefix='native-modes-cold-') as temporary:
        local=Path(temporary);shutil.copytree(directory,local/'field');shutil.copy2(out/'fixture.pt',local/'fixture.pt')
        cmd=[sys.executable,'-m','scripts.research.validate_query_field_native_modes','--child','--field',str(local/'field'),
             '--fixture',str(local/'fixture.pt'),'--output',str(local/'actual.pt')]
        for p in [repo/'output',repo/'data',Path('/root/RADIO-GS/data')]:cmd+=['--deny',str(p.resolve())]
        process=subprocess.run(cmd,cwd=local,env=dict(os.environ,PYTHONPATH=str(repo)),capture_output=True,text=True,check=True)
        log=json.loads(process.stdout.strip().splitlines()[-1]);actual=torch.load(local/'actual.pt',map_location='cpu',weights_only=True)
        shutil.copy2(local/'actual.pt',out/'copied.pt')
    comparison={}
    for m in expected:
        a,b=expected[m],actual[m]
        comparison[m]=dict(exact_scores=torch.equal(a['scores'],b['scores']),
            maximum_error=float((a['scores']-b['scores']).abs().max()),exact_selection=torch.equal(a['selected'],b['selected']),
            exact_candidates=torch.equal(a['candidates'],b['candidates']),exact_weights=torch.equal(a['weights'],b['weights']))
    strict=all(x['exact_scores'] and x['exact_selection'] and x['exact_candidates'] and x['exact_weights'] for x in comparison.values())
    numeric=all(x['maximum_error']<=1e-6 and x['exact_selection'] and x['exact_candidates'] for x in comparison.values())
    write(out/'report.json',dict(complete=True,comparison=comparison,strict_bitwise_passed=strict,
        numerical_and_selection_passed=numeric,source_denial_passed=not log['forbidden_open_attempts'],**log,
        coverage='one scene,measure_native;actual native-size image and padding plus surface2D included',
        enforcement='Python file-open guard;not OS sandbox',queries={k:v['query_id'] for k,v in queries.items()},
        evidence=[receipt(out/p) for p in ['fixture.pt','original.pt','copied.pt']]+[receipt(Path(__file__))]))
    print(comparison,flush=True)
    if not numeric or log['forbidden_open_attempts']:raise ValueError('copied field replay failed; report retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--field');p.add_argument('--fixture');p.add_argument('--output');p.add_argument('--deny',action='append',default=[])
    a=p.parse_args();torch.set_num_threads(4);child(a) if a.child else main()
