"""Copied-field replay of all figurines query fixtures, no source file access."""
import argparse,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import torch
from scripts.research.run_query_field_native_region import ROOT,SOURCE,read,write,receipt


def worker(root):
    root=Path(root);forbidden=[os.path.realpath('output'),'/mnt/pool/sqy/3d_understanding'];attempts=[]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.path.realpath(os.fsdecode(args[0]))
        if any(p==r or p.startswith(r+os.sep) for r in forbidden):attempts.append(p);raise PermissionError(p)
    sys.addaudithook(audit)
    try:open('output/denied_part_probe','rb')
    except PermissionError:probe=True
    else:raise RuntimeError('source-denial probe ineffective')
    attempts.clear()
    from radio_gs.query_field.storage import load_field
    from radio_gs.query_field.part_conditioning import prepare_parts,read_parts,native_region_probability
    field,_=load_field(root/'field','cuda');fixture=torch.load(root/'queries.pt',weights_only=True)
    queries=fixture['prior'].cuda()
    available=field.observed|field.semantic_measure.supported;outputs={}
    for acceptor in ['pooled','native']:
        for arm in [acceptor+'_parts',acceptor+'_protected']:outputs[arm]=[]
        for qi,q in enumerate(queries.T):
            probability=(native_region_probability(field.evidence,fixture['vectors'][qi].cuda(),fixture['negatives'].cuda())
                if acceptor=='native' else None)
            prepared=prepare_parts(field.evidence,q,available,region_probability=probability)
            for suffix,protect in [('parts',False),('protected',True)]:
                outputs[acceptor+'_'+suffix].append(read_parts(q,available,prepared,protect=protect)[0].cpu())
    outputs={a:torch.stack(v,1) for a,v in outputs.items()}
    from radio_gs.query_field.runtime import QueryFieldRuntime
    api=QueryFieldRuntime(field).text_union(fixture['vectors'][0].cuda(),fixture['negatives'].cuda(),
        rule='measure',temperature=10.,support_rule='native_protected_parts')
    api_error=float((api.membership_scores.cpu()-outputs['native_protected'][:,0]).abs().max())
    api_selected=torch.equal(api.selected.cpu(),outputs['native_protected'][:,0]>.5)
    torch.save(outputs,root/'replay.pt');write(root/'audit.json',dict(probe_blocked=probe,forbidden_attempts=attempts,
        public_api_max_error=api_error,public_api_selected_exact=api_selected,public_api_queries=1))


def run():
    cfg=read(ROOT/'prediction_config.json');job=cfg['jobs'][0]
    read(ROOT/'predictions'/job['name']/'prediction.json')  # Replay needs sealed predictions, never target scoring.
    out=ROOT/'validation/runtime';out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='radio_parts_') as temp:
        tmp=Path(temp);shutil.copytree(job['field'],tmp/'field')
        data=torch.load(SOURCE/'predictions'/job['name']/'predictions.pt',weights_only=True)
        torch.save(dict(prior=data['scores']['measure_prior'],vectors=data['native_query_vectors'],
            negatives=data['generic_negative_vectors']),tmp/'queries.pt');del data
        with (out/'worker.log').open('x') as f:
            subprocess.run([sys.executable,'-m','scripts.research.validate_query_field_native_region','worker','--root',str(tmp)],stdout=f,stderr=subprocess.STDOUT,check=True)
        shutil.copyfile(tmp/'replay.pt',out/'replay.pt');shutil.copyfile(tmp/'audit.json',out/'audit.json')
    expected=torch.load(ROOT/'predictions'/job['name']/'predictions.pt',weights_only=True)['scores'];actual=torch.load(out/'replay.pt',weights_only=True)
    audit=read(out/'audit.json');errors={a:float((actual[a]-expected[a]).abs().max()) for a in actual}
    exact={a:torch.equal(actual[a],expected[a]) for a in actual};selected={a:torch.equal(actual[a]>.5,expected[a]>.5) for a in actual}
    report=dict(complete=True,scene=job['scene'],queries=actual['native_parts'].shape[1],max_errors=errors,exact=exact,selected_exact=selected,
        numerical_passed=max(errors.values())<=1e-6 and all(selected.values()) and audit['public_api_max_error']<=1e-6 and audit['public_api_selected_exact'],
        public_api_max_error=audit['public_api_max_error'],public_api_selected_exact=audit['public_api_selected_exact'],
        source_denial_passed=audit['probe_blocked'] and not audit['forbidden_attempts'],
        scope='one scene/all precomputed current-query semantic fixtures, Python audit not OS isolation; raw encoding not rerun',
        code=receipt(__file__),evidence=[receipt(out/p) for p in ['worker.log','replay.pt','audit.json']])
    write(out/'report.json',report);write('docs/experiments/2026-10-10-native-region-runtime.json',report);print(report)
    if not report['numerical_passed'] or not report['source_denial_passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','worker']);p.add_argument('--root');a=p.parse_args();torch.set_num_threads(4)
    worker(a.root) if a.stage=='worker' else run()
