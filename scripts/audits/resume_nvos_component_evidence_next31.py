"""Resume an externally interrupted fixed NVOS cohort without replacing attempts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from radio_gs.utils.immutable_artifacts import sha256_file

ORIGINAL = Path('output/nvos_component_evidence_next31_v2')
OUT = Path('output/nvos_component_evidence_next31_v2_resume')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    if a.prepare:
        OUT.mkdir(exist_ok=False)
        registration = json.loads((ORIGINAL/'registration.json').read_text())
        for f, h in registration['implementations'].items():
            assert sha256_file(f) == h, f
        registration.update(original_registration_sha256=sha256_file(ORIGINAL/'registration.json'),
            resume_reason='Externally interrupted turn; original partial directories preserved',
            original_attempt_root=str(ORIGINAL))
        registration['implementations'][__file__] = sha256_file(__file__)
        (OUT/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
        adopted = {}
        for task in registration['tasks']:
            report = ORIGINAL/task/'report.json'
            if not report.exists():
                continue
            r = json.loads(report.read_text())
            assert r['baseline_mask_replay_exact']
            for variant in registration['variants']:
                assert sha256_file(ORIGINAL/task/(variant+'.pt')) == r['results'][variant]['prediction_sha256']
            shutil.copytree(ORIGINAL/task, OUT/task)
            assert sha256_file(OUT/task/'report.json') == sha256_file(report)
            adopted[task] = dict(original_report=str(report), report_sha256=sha256_file(report))
        (OUT/'adopted_completed.json').write_text(json.dumps(adopted, indent=2)+'\n')
    reg = json.loads((OUT/'registration.json').read_text())
    for f, h in reg['implementations'].items():
        assert sha256_file(f) == h, f
    for task in a.tasks or []:
        assert task in reg['tasks']
        assert not (OUT/task).exists(), 'Completed or partial attempts cannot be replaced'
        parent = Path('output/nvos_ropefix_pipeline_20260917')/('horns_center' if task=='horns_left' else task)
        cmd = [sys.executable, 'scripts/audits/evaluate_nvos_component_evidence_next31_v2.py',
               '--scene', task, '--root', str(parent/'field'), '--geometry-report',
               str(parent/'frozen_geometry/report.json'), '--runs', *reg['variants'],
               '--query-root', 'output/nvos_positive_queries_20260918', '--device', 'cuda:0',
               '--output', str(OUT/task)]
        start = time.time()
        with (OUT/(task+'.log')).open('x') as log:
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='4', PYTHONPATH='.'))
        (OUT/(task+'.execution.json')).write_text(json.dumps(dict(command=cmd,
            returncode=ret.returncode, seconds=time.time()-start), indent=2)+'\n')
        if ret.returncode:
            raise RuntimeError(task+' failed; preserve attempt')
        print(task, 'complete', flush=True)
    if a.summarize:
        reports = {s:json.loads((OUT/s/'report.json').read_text()) for s in reg['tasks']}
        summary = dict(complete=True, tasks=len(reports), development_only=True,
            registration_sha256=sha256_file(OUT/'registration.json'),
            adopted_completed_sha256=sha256_file(OUT/'adopted_completed.json'),
            macro={v:{m:sum(r['results'][v][m] for r in reports.values())/len(reports)
                for m in ['iou','pixel_accuracy','average_precision']} for v in reg['variants']},
            task_results={s:r['results'] for s,r in reports.items()},
            evidence={s:r['refinement_records']['component_evidence']['component_reference_evidence'] for s,r in reports.items()},
            baseline_masks_exact=all(r['baseline_mask_replay_exact'] for r in reports.values()),
            reports={s:sha256_file(OUT/s/'report.json') for s in reg['tasks']})
        (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary['macro'], indent=2))


if __name__ == '__main__':
    main()
