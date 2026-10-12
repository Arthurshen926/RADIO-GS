"""Resume the one failed completion worker without changing its frozen method.

This is a recovery for the recorded teatime CUDA OOM only. Completed scene
predictions stay sealed. The original controller still owns the final barrier
and both exact comparisons; only its dispatch is replaced with a missing job.
Run after the original controller and all its workers have exited.
"""
import argparse
from pathlib import Path

import torch

from scripts.research import validate_query_field_extent_completion_public as study
from scripts.research.run_query_field_relations import check, read, receipt, write


def resume(gpu):
    root = study.ROOT
    design = study.design()
    assert not (root / 'predictions_complete.json').exists()
    failure = root / 'logs/teatime.log'
    assert 'torch.cuda.OutOfMemoryError' in failure.read_text()
    preserved = []
    missing = []
    for index, job in enumerate(design['jobs']):
        path = root / 'predictions' / job['scene'] / 'result.json'
        if not path.exists():
            assert not path.parent.exists(), 'partial output requires investigation'
            missing.append(index)
            continue
        record = read(path)
        assert record['scene'] == job['scene'] and record['query'] == job['query']
        assert record['queries'] == len(job['texts'])
        assert record['query_file_io_denied'] and not record['worker_loaded_reference_predictions']
        check(job['query']); check(job['manifest']); check(record['output'])
        preserved.extend([receipt(path), record['output']])
    assert [design['jobs'][i]['scene'] for i in missing] == ['teatime']
    assert len(preserved) == 6, 'all three successful scenes must finish first'
    write(root / 'recovery_design.json', dict(
        original_design=receipt(root / 'design.json'), failure=receipt(failure),
        original_controller_log=receipt('/tmp/radio_gs_text_extent_completion_public.log'),
        preserved=preserved, retry_scene='teatime', retry_gpu=gpu,
        reason='CUDA memory exhaustion before the first query completed',
        method_changed=False, queries_changed=False, thresholds_changed=False,
        target_labels_used=False, code=receipt(__file__)))

    def missing_only(tasks, gpus, action):
        assert list(tasks) == list(range(4)) and gpus == [gpu]
        for item in preserved:
            check(item)
        action(missing[0], gpu)
        for item in preserved:
            check(item)

    original_lanes = study.lanes
    try:
        study.lanes = missing_only
        study.run([gpu])
    finally:
        study.lanes = original_lanes
    for item in preserved:
        check(item)
    result = dict(complete=True, original_outputs_unchanged=True,
                  recovery_design=receipt(root / 'recovery_design.json'),
                  retry_log=receipt(root / 'logs/teatime.attempt1.log'),
                  final_audit=receipt(root / 'audit.json'))
    write(root / 'recovery.json', result)
    write('docs/experiments/2026-10-12-text-extent-completion-public-recovery.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    resume(args.gpu)
