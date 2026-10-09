"""Copied-field signed query with explicit denied-source audit in a fresh process."""
import argparse,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import torch
from scripts.research.run_query_field_positive_rays import ROOT,read,write,receipt,design

def worker(config,output):
    forbidden=[os.path.realpath('output'),'/mnt/pool/sqy/3d_understanding'];attempts=[]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.path.realpath(os.fsdecode(args[0]))
        if any(p==r or p.startswith(r+os.sep) for r in forbidden):attempts.append(p);raise PermissionError(p)
    sys.addaudithook(audit)
    try:open('output/signed_region_denied_probe','rb')
    except PermissionError:probe=True
    else:raise RuntimeError('guard failed')
    attempts.clear()
    from radio_gs.query_field.public_nvos import predict
    c=read(config);predict(c,c['jobs'][0],output)
    write(Path(output)/'audit.json',dict(probe_blocked=probe,forbidden_open_attempts=attempts))

def run():
    d=design();read(ROOT/'scoring_complete.json');j=d['jobs'][0];cfg=read(j['configuration']['path']);job=next(x for x in cfg['jobs'] if x['query_program']=='positive_ray_local')
    out=ROOT/'validation/runtime';out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='radio_signed_region_') as temp:
        tmp=Path(temp);shutil.copytree(job['field'],tmp/'field');original=job['name'];job=dict(job,field=str(tmp/'field'))
        for k in ['positive','negative','camera_fixture']:
            dest=tmp/(k+Path(job[k]).suffix);shutil.copyfile(job[k],dest);job[k]=str(dest)
        cfg=dict(cfg,jobs=[job]);write(tmp/'config.json',cfg)
        with (out/'worker.log').open('x') as f:
            subprocess.run([sys.executable,'-m','scripts.research.validate_query_field_positive_rays','worker','--config',str(tmp/'config.json'),'--output',str(tmp/'out')],stdout=f,stderr=subprocess.STDOUT,check=True)
        shutil.copyfile(tmp/'out'/original/'predictions.pt',out/'predictions.pt');shutil.copyfile(tmp/'out/audit.json',out/'audit.json')
    expected=torch.load(ROOT/'predictions'/j['name']/original/'predictions.pt',weights_only=True);actual=torch.load(out/'predictions.pt',weights_only=True)
    differences={k:float((actual[k].float()-expected[k].float()).abs().max()) for k in actual}
    exact={k:torch.equal(actual[k],expected[k]) for k in actual};audit=read(out/'audit.json')
    report=dict(complete=True,case=j['name'],strict=all(exact.values()),exact=exact,maximum_error=differences,
        numerical_passed=exact['selected'] and max(differences.values())<=1e-6,
        source_denial_passed=audit['probe_blocked'] and not audit['forbidden_open_attempts'],
        coverage='one complete raw signed prompt case, fresh process and copied field; Python audit not OS sandbox',
        evidence=[receipt(out/p) for p in ['worker.log','predictions.pt','audit.json']]+[receipt(Path(__file__))])
    write(out/'report.json',report);write('docs/experiments/2026-10-09-positive-ray-runtime.json',report);print(report,flush=True)
    if not report['numerical_passed'] or not report['source_denial_passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','worker']);p.add_argument('--config');p.add_argument('--output');a=p.parse_args();torch.set_num_threads(4)
    run() if a.stage=='run' else worker(a.config,a.output)
