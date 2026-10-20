"""Wait for the sealed corpus/model, then evaluate and attribute both families.

The existing source/fit program remains unchanged. No labels are read by this
orchestrator, and a failed phase stops subsequent work. Logs have a hard cap.
"""
import argparse
import datetime
from pathlib import Path
import subprocess
import sys
from radio_gs.query_field.bounded_experiment import BoundedExperiment
from scripts.research.replay_tmm_response import STUDY,POLICY,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='multitask_transfer_program_v2')
    args=p.parse_args();run=BoundedExperiment(STUDY,args.run,POLICY);run.preflight(2*2**20)
    phases=[('transfer',[sys.executable,'scripts/research/evaluate_tmm_multitask_transfer.py','--wait-for-inputs']),
        ('diagnosis',[sys.executable,'scripts/research/diagnose_tmm_multitask_transfer.py']),
        ('refresh',[sys.executable,'-m','radio_gs.response_study','refresh']),
        ('verify',[sys.executable,'-m','radio_gs.response_study','verify'])]
    paths=[__file__,'scripts/research/evaluate_tmm_multitask_transfer.py','scripts/research/diagnose_tmm_multitask_transfer.py',
        'radio_gs/querying/category_response_transfer.py','radio_gs/response_study/transfer_diagnosis.py',
        'scripts/research/evaluate_tmm_response_replacement.py','radio_gs/querying/response_replacement.py',
        'radio_gs/querying/response_value.py','scripts/research/run_tmm_response_study.py']
    hashes={p:sha(p) for p in paths}
    run.json('manifest.json',dict(phases=phases,source_hashes=hashes,
        training_run='response_value_multitask_v1',prepared_run='category_transfer_inputs_v2',
        benchmark_threshold_search=False,default_changed=False,log_cap_per_phase_bytes=512*1024))
    run.json('source_snapshot.json',{p:dict(text=Path(p).read_text(),sha256=hashes[p]) for p in paths})
    receipts=[]
    for name,command in phases:
        if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('sealed transfer program source changed')
        status=dict(phase=name,status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    completed_phases=receipts)
        run.json('progress.json',status,replace=True)
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=bytearray();discarded=0
        try:
            for line in iter(process.stdout.readline,b''):
                print(line.decode(errors='replace'),end='',flush=True)
                space=max(0,512*1024-len(log));log.extend(line[:space]);discarded+=max(0,len(line)-space)
            code=process.wait()
        finally:
            if process.poll() is None:process.terminate();process.wait()
            process.stdout.close()
        run._write(name+'.log',lambda f:f.write(log),limit=max(1,len(log)))
        receipt=dict(phase=name,exit_code=code,log_bytes=len(log),discarded_log_bytes=discarded)
        run.json(name+'_receipt.json',receipt);receipts.append(receipt)
        if code:
            run.json('progress.json',dict(status,status='failed',completed_phases=receipts),replace=True)
            raise SystemExit(code)
    run.json('summary.json',dict(complete=True,phases=receipts,default_changed=False))
    run.json('progress.json',dict(status='complete',completed_phases=receipts),replace=True)


if __name__=='__main__':main()
