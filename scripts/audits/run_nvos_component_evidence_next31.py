"""One registered reference-evidence rule, eight tasks, unchanged scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from radio_gs.utils.immutable_artifacts import sha256_file

OUT = Path('output/nvos_component_evidence_next31')
TASKS = ['flower', 'fortress', 'fern', 'horns_center', 'horns_left', 'leaves', 'orchids', 'trex']
SOURCES = [__file__, 'scripts/audits/evaluate_nvos_component_evidence_next31.py',
           'scripts/audits/nvos_component_evidence_runtime_next31.py',
           'radio_gs/querying/component_reference_evidence.py']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--register', action='store_true')
    p.add_argument('--tasks', nargs='+', choices=TASKS)
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    if a.register:
        OUT.mkdir(exist_ok=False)
        reg = dict(schema='nvos_component_evidence_next31.registration.v1', tasks=TASKS,
            variants=['baseline', 'component_evidence'], baseline='reference_core_component_next19',
            method='preserve seeded components; additionally retain whole positive spatial components whose actual full-reference eroded-SAM adjoint mass has purity >=0.9 and foreground mass >=1 effective pixel',
            purity=.9, minimum_reference_pixel_mass=1., erosion_width=5,
            k=16, radius_multiplier=3., unchanged_positive_repair=True,
            reference_sam_disconnected_pieces_are_teacher_hypotheses=True,
            missing_visibility_is_not_positive_identity=True, no_new_negative_labels=True,
            negative_scribbles_allowed=False, target_rgb_allowed=False,
            target_gt_used_after_prediction_only=True, training_feature_reads=False,
            target_parameter_search=False, full_method=False, default_promoted=False,
            implementations={f: sha256_file(f) for f in SOURCES})
        (OUT/'registration.json').write_text(json.dumps(reg, indent=2)+'\n')
    reg = json.loads((OUT/'registration.json').read_text())
    for f, digest in reg['implementations'].items():
        assert sha256_file(f) == digest, f
    for task in a.tasks or []:
        parent = Path('output/nvos_ropefix_pipeline_20260917')/('horns_center' if task=='horns_left' else task)
        cmd = [sys.executable, 'scripts/audits/evaluate_nvos_component_evidence_next31.py',
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
            raise RuntimeError(task+' failed; preserve attempt and inspect log')
        print(task, 'complete', flush=True)
    if a.summarize:
        reports = {s: json.loads((OUT/s/'report.json').read_text()) for s in TASKS}
        result = dict(registration_sha256=sha256_file(OUT/'registration.json'), complete=True,
            macro={v: {m:sum(r['results'][v][m] for r in reports.values())/len(TASKS)
                for m in ['iou', 'pixel_accuracy', 'average_precision']} for v in reg['variants']},
            task_results={s:r['results'] for s,r in reports.items()},
            evidence={s:r['refinement_records']['component_evidence']['component_reference_evidence'] for s,r in reports.items()},
            baseline_masks_exact=all(r['baseline_mask_replay_exact'] for r in reports.values()),
            reports={s:sha256_file(OUT/s/'report.json') for s in TASKS})
        (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['macro'], indent=2))


if __name__ == '__main__':
    main()
