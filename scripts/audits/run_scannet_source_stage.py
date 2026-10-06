"""Deny semantic labels and cached benchmark features during ScanNet source training."""
import argparse
import os
from pathlib import Path
import runpy
import sys

p=argparse.ArgumentParser();p.add_argument('--module',action='store_true');p.add_argument('entry');a,rest=p.parse_known_args()
forbidden=Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label').resolve()
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path==forbidden or forbidden in path.parents or any(x in str(path) for x in ('.labels.ply','language_features','instance_annotations','/label-filt/','/instance-filt/')):raise RuntimeError('Benchmark labels/features forbidden during source reconstruction')
sys.addaudithook(guard);sys.argv=[a.entry,*rest]
if a.module:runpy.run_module(a.entry,run_name='__main__')
else:runpy.run_path(a.entry,run_name='__main__')
