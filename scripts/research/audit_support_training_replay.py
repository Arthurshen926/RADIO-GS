"""Repeat the exact pre-study fitter to distinguish training from query replay."""
import hashlib
import json
from pathlib import Path
import subprocess
import types

import torch


def main():
    torch.set_num_threads(4)
    root = Path('output/query_field_20261007/support_training_v1')
    modules = {}
    receipts = []
    for name in ['field', 'train']:
        source = subprocess.check_output(['git', 'show', f'b831d03:radio_gs/query_field/{name}.py'])
        path = root/f'original_{name}.py'
        path.write_bytes(source)
        receipts.append(dict(path=str(path), sha256=hashlib.sha256(source).hexdigest(), revision='b831d03'))
        module = types.ModuleType(f'radio_gs.query_field.original_{name}')
        module.__package__ = 'radio_gs.query_field'
        exec(compile(source, str(path), 'exec'), module.__dict__)
        modules[name] = module
    modules['train'].CapabilityField = modules['field'].CapabilityField
    (root/'original_replay_sources.json').write_text(json.dumps(receipts, indent=2)+'\n')
    config = json.loads(Path('configs/query_field/nvos_fern_physical.json').read_text())['training']
    for repeat in [1, 2]:
        out = root/f'fern_original_replay_{repeat}'
        out.mkdir(exist_ok=False)
        observation = Path('output/query_field_20261007/nvos_fern_v1/observations').resolve()
        (out/'observations').symlink_to(observation, target_is_directory=True)
        report = modules['train'].fit(observation, out/'models', config)
        print('original repeat complete', repeat, report['seconds'], flush=True)


if __name__ == '__main__':
    main()
