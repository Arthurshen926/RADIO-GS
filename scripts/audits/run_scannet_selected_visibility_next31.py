"""Preregistered paired selected-scope controls on all eight frozen ScanNet maps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from radio_gs.utils.immutable_artifacts import sha256_file


OUTPUT=Path('output/scannet_selected_visibility_next31')
TRAINER=Path('scripts/audits/train_registered_selected_scope_next31.py')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true')
    parser.add_argument('--scenes',nargs='+');args=parser.parse_args()
    if args.register:
        OUTPUT.mkdir(exist_ok=False)
        prior=json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())
        rows=[]
        for row in prior['cohort']:
            if row['benchmark']!='scannet':continue
            field=Path(row['root']);scene=row['key'].removeprefix('scannet_')
            geometry=field.parent.parent/'geometry_report.json'
            report=field/'bounded_fallback_admission/report.json'
            receipt=json.loads(report.read_text())
            assert sha256_file(report.with_name('field.pt'))==receipt['field_sha256']
            rows.append(dict(scene=scene,root=str(field),geometry_report=str(geometry),
                geometry_report_sha256=sha256_file(geometry),parent_field_sha256=receipt['field_sha256'],
                parent_report_sha256=sha256_file(report),teacher_manifest_sha256=sha256_file(field/'teachers.json'),
                authority_sha256=sha256_file(field/'responsibility.json')))
        assert len(rows)==8
        reg=dict(schema='scannet_selected_visibility_paired.next31',scenes=rows,
            parent_name='bounded_fallback_admission',modes=['selected','visible_selected'],steps=400,
            source_only_training=True,source_rule='Same frozen LERF objective and joint full/selected checkpoint acceptance; only geometry camera provider changes to bound responsibility metadata.',
            evaluation_rule='All eight scenes and both fixed arms, retain old source scope/admission policy including0590rejection; calibrated native + boundedscope correction max1. No target branch selection. Native and retained source-admitted arrays must replay exactly before GT.',
            source_validation='Last four source training views, held out from new head fit only; parent-seen, fixed order and first alias. Step zero eligible.',
            implementation_sha256=sha256_file(TRAINER),visibility_implementation_sha256=sha256_file('radio_gs/training/selected_output_supervision.py'),
            driver_sha256=sha256_file(__file__))
        (OUTPUT/'registration.json').write_text(json.dumps(reg,indent=2)+'\n')
        print('registered8',flush=True);return
    reg=json.loads((OUTPUT/'registration.json').read_text())
    assert sha256_file(TRAINER)==reg['implementation_sha256']
    assert sha256_file('radio_gs/training/selected_output_supervision.py')==reg['visibility_implementation_sha256']
    for row in reg['scenes']:
        if row['scene'] not in args.scenes:continue
        output=OUTPUT/row['scene'];output.mkdir(exist_ok=True)
        root=Path(row['root'])
        assert sha256_file(root/reg['parent_name']/'field.pt')==row['parent_field_sha256']
        assert sha256_file(row['geometry_report'])==row['geometry_report_sha256']
        for mode in reg['modes']:
            receipt=output/(mode+'.execution.json')
            cmd=[sys.executable,'scripts/audits/run_scannet_source_stage.py',str(TRAINER),
                 '--root',str(root),'--geometry-report',row['geometry_report'],'--device','cuda:0',
                 '--mode',mode,'--steps',str(reg['steps']),'--parent-name',reg['parent_name'],
                 '--output',str(output/mode)]
            if receipt.exists():
                previous=json.loads(receipt.read_text());assert previous['status']=='complete' and previous['command']==cmd
                assert sha256_file(output/mode/'report.json')==previous['report_sha256'];continue
            record=dict(status='running',command=cmd,registration_sha256=sha256_file(OUTPUT/'registration.json'))
            receipt.write_text(json.dumps(record,indent=2)+'\n')
            with (output/(mode+'.log')).open('x') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',PYTHONPATH='.'))
            record.update(status='failed' if result.returncode else 'complete',returncode=result.returncode)
            if not result.returncode:record['report_sha256']=sha256_file(output/mode/'report.json')
            receipt.write_text(json.dumps(record,indent=2)+'\n')
            if result.returncode:raise RuntimeError(str(receipt))
            print(row['scene'],mode,'complete',flush=True)


if __name__=='__main__':main()
