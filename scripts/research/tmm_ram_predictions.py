"""Bounded volatile prediction bank on Linux tmpfs, recoverable after a crash.

Only private /dev/shm directories are accepted. No prediction arrays are written
to persistent storage. Completed scoring removes this run's bank explicitly.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from scripts.research.run_tmm_support_agreement import array_sha, RAM_LIMIT


class RamPredictionBank:
    def __init__(self, directory=None, limit=RAM_LIMIT):
        mounts=[line.split() for line in Path('/proc/mounts').read_text().splitlines()]
        if not any(row[1]=='/dev/shm' and row[2]=='tmpfs' for row in mounts):
            raise ValueError('prediction bank requires /dev/shm mounted as tmpfs')
        self.root=Path(directory or tempfile.mkdtemp(prefix='radiogs_tmm_',dir='/dev/shm')).absolute()
        if self.root.is_symlink() or self.root.resolve().parent!=Path('/dev/shm'):
            raise ValueError('bank must be a private direct child of /dev/shm')
        self.root.mkdir(mode=0o700,exist_ok=True)
        if self.root.stat().st_uid!=os.getuid() or self.root.stat().st_mode & 0o077:
            raise ValueError('prediction bank must be private to the current user')
        self.limit=limit;self.manifest=self.root/'manifest.json'
        self.entries=json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        self.bytes=sum(v['bytes'] for v in self.entries.values())
        if self.bytes>limit:raise ValueError('existing prediction bank exceeds RAM budget')

    def key(self,run,arm):return str(run.root)+'::'+arm

    def add(self,run,arm,value):
        key=self.key(run,arm)
        if key in self.entries:raise ValueError('duplicate RAM-bank prediction')
        array=np.ascontiguousarray(value.cpu().numpy(),dtype=np.float32)
        # Include NPY headers and manifest overhead in the hard tmpfs bound.
        used=sum(p.stat().st_size for p in self.root.iterdir())
        if used+array.nbytes+4096*(len(self.entries)+1)>self.limit:
            raise ValueError('transient prediction RAM budget exceeded')
        name=hashlib.sha256(key.encode()).hexdigest()+'.npy';path=self.root/name
        with (self.root/'pending.npy').open('wb') as handle:np.save(handle,array,allow_pickle=False)
        os.replace(self.root/'pending.npy',path)
        record=dict(shape=list(array.shape),dtype='float32',sha256=array_sha(array),bytes=array.nbytes)
        self.entries[key]=dict(record,file=name);self.bytes+=array.nbytes
        temporary=self.root/'pending.json';temporary.write_text(json.dumps(self.entries))
        os.replace(temporary,self.manifest)
        return record

    def read(self,run,arm,count):
        record=self.entries[self.key(run,arm)]
        array=np.load(self.root/record['file'],mmap_mode='r',allow_pickle=False)
        if (array.dtype!=np.float32 or list(array.shape)!=record['shape'] or
                array.shape[1]!=count or array_sha(array)!=record['sha256']):
            raise ValueError('sealed RAM prediction changed')
        return torch.from_numpy(array.copy())

    def discard_unsealed(self,run):
        """Caller may use this only when the job has no prediction report."""
        for key in list(self.entries):
            if key.startswith(str(run.root)+'::'):
                record=self.entries.pop(key);(self.root/record['file']).unlink(missing_ok=True)
                self.bytes-=record['bytes']
        temporary=self.root/'pending.json';temporary.write_text(json.dumps(self.entries))
        os.replace(temporary,self.manifest)

    def close(self):
        # Delete only files explicitly owned by this bank, never another run.
        for record in self.entries.values():(self.root/record['file']).unlink()
        self.manifest.unlink(missing_ok=True)
        for name in ['pending.npy','pending.json']:(self.root/name).unlink(missing_ok=True)
        self.root.rmdir()


def public_query_files(path):
    path=Path(path)
    files=[path] if path.is_file() else [path/'queries'/f'{m}.json'
        for m in ['text','image','point_2d','point_3d']]
    if not all(p.is_file() for p in files):raise ValueError('public query files are missing')
    return files
