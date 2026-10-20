"""Pipeline receipts bind fresh scored outputs before opening target labels."""
import hashlib
from pathlib import Path


def prediction_seals(root, recipe):
    root=Path(root);seals={}
    for job in recipe['jobs']:
        base=root/job['benchmark']/job['name']/'query'
        if job['benchmark']=='uqis':
            paths=[]
            for readout in job.get('readouts',[job['primary_readout']]):
                prediction=base/'registered'/readout/'predictions'
                selected=prediction.parent/'selections'
                scores=sorted(prediction.glob('*.npy'))
                if not scores:raise ValueError('fresh predictions missing for '+job['name']+'/'+readout)
                selections=sorted(selected.glob('*.npy'))
                if selected.exists() and {p.stem for p in scores}!={p.stem for p in selections}:
                    raise ValueError('query scores and explicit selections differ')
                paths.extend(scores+selections)
        else:
            paths=[base/('category/typed_readout.pt' if job['benchmark']=='scannet' else 'typed_readout.pt')]
        for path in paths:
            h=hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(2**20),b''):h.update(block)
            seals[str(path.relative_to(root))]=dict(sha256=h.hexdigest(),bytes=path.stat().st_size)
    return seals
