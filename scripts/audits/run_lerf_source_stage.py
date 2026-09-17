"""Deny real evaluation labels while executing a LERF reconstruction stage."""
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
        if path==forbidden or forbidden in path.parents:raise RuntimeError('LERF real labels forbidden during source reconstruction')
sys.addaudithook(guard);sys.argv=[a.entry,*rest]
if a.module:runpy.run_module(a.entry,run_name='__main__')
else:runpy.run_path(a.entry,run_name='__main__')
