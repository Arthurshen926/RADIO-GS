"""Bounded non-destructive pilot. Never deletes or replaces any existing file."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from radio_gs.utils.inference_checkpoint import (
    DROP_KEYS, fingerprint, load_inference_checkpoint, make_inference_package,
    read_project_checkpoint_once,
)
from scripts.audits.inference_pilot_probes import probes

ROOT = Path('/mnt/pool/sqy/results/RADIO-GS/output')
DEST = ROOT / 'storage_inference_pilot_20261014_1146'
RUNS = [
 ('scannet_hybrid', 'scannet_og_scene0000_00_v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63'),
 ('lerf_explicit', 'lerf_figurines_component_no_hybrid_seed7'),
 ('vpr_adapter', 'vpr_field_consistency_20260515/figurines'),
]
MAX_WRITE = 1024**3
RAM_LIMIT = 3 * 1024**3
written = 0

def dump_new(path, data):
    global written
    payload = (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if written + len(payload) > MAX_WRITE:
        raise RuntimeError('new-output budget exceeded')
    with open(path, 'xb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    written += len(payload)

class NewWriter:
    def __init__(self, handle):
        self.handle = handle; self.hash = hashlib.sha256(); self.count = 0
    def write(self, b):
        global written
        if written + len(b) > MAX_WRITE:
            raise RuntimeError('new-output budget exceeded')
        n = self.handle.write(b)
        if n != len(b):
            raise RuntimeError('short write')
        self.hash.update(b); self.count += n; written += n
        return n
    def flush(self): self.handle.flush()
    def tell(self): return self.handle.tell()

def tensor_bytes(t):
    return memoryview(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())

def compare(a, b, path='', tensors=None):
    if tensors is None: tensors = []
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride(), path
        aa, bb = tensor_bytes(a), tensor_bytes(b)
        assert aa == bb, 'tensor bytes differ: '+path
        tensors.append({'path':path,'shape':list(a.shape),'dtype':str(a.dtype),
                        'logical_bytes':a.numel()*a.element_size(),
                        'sha256':hashlib.sha256(aa).hexdigest()})
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), path
        for k in a: compare(a[k], b[k], path+'/'+str(k), tensors)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a)==len(b), path
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'/'+str(i),tensors)
    elif isinstance(a, np.generic):
        assert type(a) is type(b) and a.tobytes()==b.tobytes(), path
    elif isinstance(a, float):
        assert isinstance(b,float) and struct.pack('d',a)==struct.pack('d',b),path
    else:
        assert type(a) is type(b) and a == b, path
    return tensors

def alias_groups(value):
    groups={}
    def walk(x,p):
        if isinstance(x,torch.Tensor):
            s=x.untyped_storage();groups.setdefault((s.data_ptr(),s.nbytes()),[]).append(p)
        elif isinstance(x,dict):
            for k,v in x.items():walk(v,p+'/'+str(k))
        elif isinstance(x,(tuple,list)):
            for i,v in enumerate(x):walk(v,p+'/'+str(i))
    walk(value,'')
    return sorted(sorted(v) for v in groups.values())

def runtime_sources(repo):
    paths={Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if raw:
            p=Path(raw).resolve()
            if str(p).startswith(str(repo)+'/') and p.suffix=='.py':paths.add(p)
    paths.add(repo/'radio_gs/training/artifact_mixin.py')
    paths.add(repo/'radio_gs/scripts/run_repo_python.sh')
    records=[]
    for p in sorted(paths):
        raw=p.read_bytes();assert len(raw)<4*1024**2
        records.append({'path':str(p.relative_to(repo)),'sha256':hashlib.sha256(raw).hexdigest(),
                        'text':raw.decode('utf-8')})
    assert sum(len(x['text']) for x in records)<32*1024**2
    return records

def self_test():
    c={'model_state_dict':{'w':torch.tensor([1.,-0.])},'codec_state_dict':{'w':torch.arange(3)},
       'optimizer_state_dict':{'a':torch.ones(2)},'scheduler_state_dict':{},'scaler_state_dict':{},'epoch':3}
    p=make_inference_package(c,{'config':{}})
    assert not DROP_KEYS.intersection(p['inference_state'])
    assert 'model_state_dict' not in p
    compare({k:v for k,v in c.items() if k not in DROP_KEYS},p['inference_state'])
    try: load_inference_checkpoint('/nonexistent',expected_sha256='0'*64,resume=True)
    except ValueError as e:assert 'cannot resume' in str(e)
    else:raise AssertionError('resume accepted')
    print('PASS in-memory envelope/preservation/resume contract')

def main():
    global written
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available()
    repo=Path.cwd().resolve()
    import importlib
    for name in ['radio_gs.models.explicit_gaussian','radio_gs.models.hybrid_gaussian',
                 'radio_gs.models.hcd_codec','radio_gs.models.point_summary_adapter',
                 'radio_gs.models.featsharp_3d','radio_gs.models.screen_refiner',
                 'radio_gs.heads.depth_head','radio_gs.heads.segmentation_head']:
        importlib.import_module(name)
    sources=runtime_sources(repo)
    planned=[]
    for name,run in RUNS:
        path=ROOT/'radio_gs'/run/'checkpoints/best.pth';fp=fingerprint(path.stat())
        assert fp['nlink']==1 and fp['size']<800_000_000
        cfgpath=path.parent.parent/'reports/resolved_config.json'
        cfgrecord=None
        if cfgpath.exists():
            raw=cfgpath.read_bytes()
            cfgrecord={'path':str(cfgpath),'sha256':hashlib.sha256(raw).hexdigest(),
                       'config':json.loads(raw)}
        planned.append({'name':name,'path':str(path),'fingerprint':fp,'config_record':cfgrecord})
    free=shutil.disk_usage(ROOT).free
    assert free>20*MAX_WRITE, 'insufficient safety headroom'
    # A pre-existing destination is a hard error; no overwrite/reuse fallback.
    DEST.mkdir(exist_ok=False)
    pre={'schema':'radiogs.inference_pilot.v1','status':'prepared_originals_retained',
         'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'source_git_status':subprocess.check_output(['git','status','--short'],text=True),
         'python':sys.version,'torch':torch.__version__,'cuda_available':False,
         'disk_free_before':free,'max_new_bytes':MAX_WRITE,'max_rss_bytes':RAM_LIMIT,
         'plan':planned,'runtime_sources':sources,
         'writer_visibility':'No visible training process in this container; remote writers cannot be globally excluded. No original checkpoint or scientific path is modified.'}
    dump_new(DEST/'preflight.json',pre)
    stop=threading.Event()
    def guard():
        while not stop.wait(0.05):
            rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
            if rss>RAM_LIMIT:
                try:dump_new(DEST/'memory_limit_failure.json',{'rss_bytes':rss,'status':'aborted; originals unchanged; new files unaccepted'})
                finally:os._exit(90)
    thread=threading.Thread(target=guard,daemon=True);thread.start()
    results=[];start=time.monotonic()
    try:
        for item in planned:
            original,record=read_project_checkpoint_once(item['path'],expected_fingerprint=item['fingerprint'])
            cfgrecord=item['config_record']
            if cfgrecord is None:
                cfg=original.get('point_summary_adapter_metadata',{}).get('config_snapshot')
                assert isinstance(cfg,dict), 'no recoverable configuration'
                cfgrecord={'path':'embedded:point_summary_adapter_metadata.config_snapshot','config':cfg}
            cfg=cfgrecord['config']
            expected={k:v for k,v in original.items() if k not in DROP_KEYS}
            before_outputs=probes(expected,cfg)
            provenance={'source_checkpoint':record,'config_record':cfgrecord,
                        'source_head':pre['source_head'],
                        'runtime_source_sha256':{x['path']:x['sha256'] for x in sources},
                        'recovery_limit':'No exact optimizer/scheduler/scaler resume from this package. Original is retained unchanged.',
                        'scope':'CPU contract and deterministic numerical probes only; not a full benchmark/render.'}
            package=make_inference_package(original,provenance)
            out=DEST/(item['name']+'.inference.pt')
            with open(out,'xb') as f:
                writer=NewWriter(f);torch.save(package,writer);writer.flush();os.fsync(f.fileno())
            digest=writer.hash.hexdigest()
            loaded,newprov,newrecord=load_inference_checkpoint(out,expected_sha256=digest)
            tensors=compare(expected,loaded)
            assert alias_groups(expected)==alias_groups(loaded), 'storage alias topology differs'
            assert newprov==provenance
            after_outputs=probes(loaded,cfg)
            checks=compare(before_outputs,after_outputs)
            assert fingerprint(os.stat(item['path'],follow_symlinks=False))==item['fingerprint']
            # Real old trainer is never called in resume mode; the raw envelope
            # lacks its required top-level model_state_dict and therefore fails
            # before parameter/epoch state can be loaded. The dedicated API rejects.
            try:load_inference_checkpoint(out,expected_sha256=digest,resume=True)
            except ValueError as e:assert 'cannot resume' in str(e)
            else:raise AssertionError('resume accepted')
            try:
                with open(out,'xb'):pass
            except FileExistsError:pass
            else:raise AssertionError('no-overwrite failed')
            row={'name':item['name'],'source':record,'package':newrecord,
                 'removed_keys':package['removed_training_keys'],'all_non_training_values_equal':True,
                 'tensor_count':len(tensors),'tensor_manifest':tensors,
                 'probe_count':len(checks),'probes':checks,'resume_rejected':True,
                 'no_overwrite_verified':True,'alias_topology_equal':True,
                 'original_stat_unchanged':True,'source_bytes':record['bytes_read'],
                 'new_logical_bytes':writer.count,'new_allocated_bytes':out.stat().st_blocks*512}
            dump_new(DEST/(item['name']+'.acceptance.json'),row);results.append(row)
            print(json.dumps({k:row[k] for k in ['name','tensor_count','probe_count','new_logical_bytes','new_allocated_bytes']}),flush=True)
            del original,expected,loaded,package,before_outputs,after_outputs,newprov,tensors,checks
            gc.collect()
        assert runtime_sources(repo)==sources,'runtime sources changed during pilot'
        for item in planned:
            assert fingerprint(os.stat(item['path'],follow_symlinks=False))==item['fingerprint']
        maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert maxrss<=RAM_LIMIT
        final={'status':'CPU_PILOT_ACCEPTED_ORIGINALS_RETAINED','scope':'CPU contracts and deterministic small numerical probes, NOT full benchmark/renderer acceptance',
               'file_count':len(results),'source_bytes_read':sum(r['source_bytes'] for r in results),
               'new_package_bytes_read':sum(r['new_logical_bytes'] for r in results),
               'new_package_logical_bytes':sum(r['new_logical_bytes'] for r in results),
               'new_package_allocated_bytes':sum(r['new_allocated_bytes'] for r in results),
               'maxrss_bytes':maxrss,'elapsed_seconds':time.monotonic()-start,
               'originals_retained':True,'actual_reclaimed_bytes':0,
               'runtime_sources_unchanged':True,'acceptance_files':[r['name']+'.acceptance.json' for r in results]}
        dump_new(DEST/'completion.json',final);print(json.dumps(final),flush=True)
    except Exception as e:
        dump_new(DEST/'failure.json',{'status':'FAILED_NEW_OUTPUTS_UNACCEPTED_ORIGINALS_RETAINED','error':repr(e),'accepted_before_failure':[r['name'] for r in results]})
        raise
    finally:stop.set()

if __name__=='__main__':
    if '--self-test' in sys.argv:self_test()
    else:main()
