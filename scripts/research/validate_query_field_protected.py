"""Real copied-export cold replay for all three protected-native arms."""
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from scripts.research.run_query_field_protected import ROOT, PREVIOUS, ARMS, read, write, receipt, check
from scripts.research.validate_query_field_program import results


def main(*, store_root=None):
    root = ROOT / 'validation/runtime'; root.mkdir(parents=True, exist_ok=False)
    prior = read(PREVIOUS / 'validation/runtime/report.json')
    check(prior['fixture'])
    fixture = torch.load(prior['fixture']['path'], map_location='cpu', weights_only=True)
    repo = Path.cwd().resolve()
    forbidden = [repo / 'output', repo / 'data', Path('/root/RADIO-GS/output'), Path('/root/RADIO-GS/data')]
    reports = []
    for arm in ARMS:
        original = ROOT / 'fits' / arm / 'uqis_scene0030_00/field'
        expected = results(original, fixture, 'common_program',store_root=store_root)
        gc.collect(); torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix='query-field-protected-cold-') as temporary:
            isolated = Path(temporary)
            if read(original/'manifest.json')['schema'].endswith('.v2'):
                from radio_gs.query_field.shared_storage import export_field_v1
                export_field_v1(original,isolated/'scene',store_root=store_root)
            else:shutil.copytree(original, isolated / 'scene')
            shutil.copy2(prior['fixture']['path'], isolated / 'query.pt')
            command = [sys.executable, '-m', 'scripts.research.validate_query_field_program', '--child',
                       '--mode', 'common_program', '--field', str(isolated / 'scene'), '--fixture', str(isolated / 'query.pt')]
            for path in forbidden:
                command.extend(['--deny', str(path.resolve())])
            process = subprocess.run(command, cwd=isolated, env=dict(os.environ, PYTHONPATH=str(repo)),
                                     capture_output=True, text=True, check=True)
            cold = json.loads(process.stdout.strip().splitlines()[-1])
        parity = {name: all(expected[name][k] == cold['results'][name][k]
                           for k in ['scores_sha256', 'selected_sha256', 'selected']) for name in expected}
        write(root / (arm + '.comparison.json'), dict(expected=expected, copied=cold, exact_replay=parity))
        if not all(parity.values()) or cold['forbidden_open_attempts']:
            raise ValueError('copied-export replay or source denial failed')
        reports.append(dict(arm=arm, field=(receipt(original/'field.pt') if (original/'field.pt').exists() else __import__('radio_gs.query_field.research_storage',fromlist=['method_binding']).method_binding(original)), exact_replay=parity, **cold))
        print('cold replay complete', arm, flush=True)
    write(root / 'report.json', dict(records=reports, fixture=prior['fixture'],
        code=[receipt(Path(__file__)), receipt(Path('scripts/research/validate_query_field_program.py'))],
        copied_export=True, fresh_process=True, benchmark_result=False,
        enforcement='Python file-open audit; not an OS security sandbox',
        coverage='one real scene per arm, four input types and a three-category current-query menu'))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
