"""Bounded sequential completion: full-teacher corpus, joint fit, study refresh.

No benchmark inference or default promotion. A failed phase stops the program.
Logs are capped in RAM and persisted through the study storage budget.
"""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.replay_tmm_response import STUDY, POLICY, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='category_multitask_program_v1')
    parser.add_argument('--sam-gpu',default='2'); args=parser.parse_args()
    run=BoundedExperiment(STUDY,args.run,POLICY); run.preflight(2*2**20)
    phases=[('source', [sys.executable,'scripts/research/compile_tmm_category_utility.py','--resume','--sam-gpu',args.sam_gpu]),
        ('fit',[sys.executable,'scripts/research/train_tmm_multitask_value.py']),
        ('refresh',[sys.executable,'-m','radio_gs.response_study','refresh']),
        ('verify',[sys.executable,'-m','radio_gs.response_study','verify'])]
    paths=[__file__, 'scripts/research/compile_tmm_category_utility.py','scripts/research/stream_tmm_category_detections.py',
        'scripts/research/train_tmm_multitask_value.py','configs/research/response_value_multitask.json',
        'radio_gs/querying/multitask_response_value.py','radio_gs/querying/category_response_supervision.py']
    hashes={p:sha(p) for p in paths}
    run.json('manifest.json',dict(phases=phases,source_hashes=hashes,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),sam_gpu=args.sam_gpu,
        benchmark_inference=False, default_changed=False, log_cap_per_phase_bytes=512*1024))
    run.json('source_snapshot.json',{p:dict(text=Path(p).read_text(),sha256=hashes[p]) for p in paths})
    receipts=[]
    for name,command in phases:
        if any(sha(p)!=h for p,h in hashes.items()):
            raise ValueError('sealed program source changed while running')
        status=dict(phase=name,status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_phases=receipts)
        run.json('progress.json',status,replace=True)
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=bytearray(); discarded=0
        try:
            for line in iter(process.stdout.readline,b''):
                print(line.decode(errors='replace'),end='',flush=True)
                space=max(0,512*1024-len(log)); log.extend(line[:space]); discarded+=max(0,len(line)-space)
            code=process.wait()
        finally:
            if process.poll() is None:
                process.terminate(); process.wait()
            process.stdout.close()
        run._write(name+'.log',lambda f:f.write(log),limit=max(1,len(log)))
        receipt=dict(phase=name,exit_code=code,log_bytes=len(log),discarded_log_bytes=discarded)
        run.json(name+'_receipt.json',receipt); receipts.append(receipt)
        if code:
            run.json('progress.json',dict(status, status='failed',completed_phases=receipts),replace=True)
            raise SystemExit(code)
    run.json('summary.json',dict(complete=True,phases=receipts,benchmark_inference=False,default_changed=False))
    run.json('progress.json',dict(status='complete',completed_phases=receipts),replace=True)


if __name__=='__main__':
    main()
