"""Matched frozen stage diagnostics; no model selection or updates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scene', required=True); args = parser.parse_args()
    root = Path('output/depth_opacity_pipeline_20260918')/args.scene
    assert json.loads((root/'execution.json').read_text())['complete']
    out = root/'stage_diagnostics'; out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out/Path(__file__).name)
    runs = ['scale_affinity_v1_direct_control', 'source_precision_gate96']
    records = {}
    for arm in ['before', 'after']:
        field = Path('output/v2_scope_rebuild_20260918')/args.scene/'field' if arm == 'before' else root/'reconstruction/field'
        gp = Path('output/geometry_only_repair_final_20260914')/args.scene/'report.json' if arm == 'before' else Path('output/depth_opacity_repair_20260918')/args.scene/'geometry_report.json'
        assert json.loads((field/'native_query_encoder.json').read_text()) == json.loads((root/'reconstruction/field/native_query_encoder.json').read_text())
        dest = out/(arm+'.json')
        command = [sys.executable, 'scripts/audits/audit_current_query_routes.py', '--task', 'text', '--root', str(field), '--scene', args.scene, '--geometry-report', str(gp), '--runs', *runs, '--device', 'cuda:0', '--encoding-replay', str(root/'text_online.encodings.pt'), '--output', str(dest)]
        with (out/(arm+'.log')).open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='4'))
        records[arm] = dict(command=command, output_sha256=sha256_file(dest), runs=json.loads(dest.read_text())['runs'])
    (out/'report.json').write_text(json.dumps(dict(scene=args.scene, source_model_updates=False, diagnostic_development=True, records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
