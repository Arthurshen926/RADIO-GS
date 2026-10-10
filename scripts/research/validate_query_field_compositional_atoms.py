"""Cold atom decoder replay with copied privileged query fixtures, not deployment."""
import argparse,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import torch
from scripts.research.run_query_field_compositional_atoms import ROOT,SOURCE,read,write,receipt,check,design


def worker(root):
    root=Path(root);forbidden=[os.path.realpath('output'),'/mnt/pool/sqy/3d_understanding'];attempts=[]
    def deny(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.path.realpath(os.fsdecode(args[0]))
        if any(p==r or p.startswith(r+os.sep) for r in forbidden):attempts.append(p);raise PermissionError(p)
    sys.addaudithook(deny)
    try:open('output/denied_atom_probe','rb')
    except PermissionError:probe=True
    else:raise RuntimeError('source denial ineffective')
    attempts.clear()
    from radio_gs.query_field.compositional_evidence import CompositionalEvidence
    from radio_gs.query_field.part_conditioning import read_parts
    model=CompositionalEvidence(torch.load(root/'atoms.pt',weights_only=True),'cuda')
    q=torch.load(root/'queries.pt',weights_only=True);available=q['available'].cuda();result={a:[] for a in ['atom_parts','atom_protected']}
    for i,prior in enumerate(q['prior'].T):
        prepared=model.prepare(q['accepted'][:,i].cuda())
        for a,protect in [('atom_parts',False),('atom_protected',True)]:
            result[a].append(read_parts(prior.cuda(),available,prepared,protect=protect)[0].cpu())
    torch.save({a:torch.stack(v,1) for a,v in result.items()},root/'replay.pt')
    write(root/'audit.json',dict(probe_blocked=probe,forbidden_attempts=attempts))


def run():
    from radio_gs.query_field.storage import load_field
    d=design();j=read(d['configuration']['path'])['jobs'][0];p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
    check(r['compilation']);check(read(r['compilation']['path'])['artifact']);check(r['teacher_selection'])
    out=ROOT/'validation/runtime';out.mkdir(parents=True,exist_ok=False)
    field,_=load_field(j['field']);available=field.observed|field.semantic_measure.supported
    mass=torch.sparse.mm(field.evidence.matrix('positive').transpose(0,1),available.float()[:,None]).flatten()
    teacher=torch.load(r['teacher_selection']['path'],weights_only=True)
    data=torch.load(SOURCE/'predictions'/j['name']/'predictions.pt',weights_only=True)
    with tempfile.TemporaryDirectory(prefix='radio_atoms_') as temp:
        tmp=Path(temp);shutil.copyfile(p/'atoms.pt',tmp/'atoms.pt')
        torch.save(dict(prior=data['scores']['measure_prior'],available=available,accepted=(teacher['probability']>.5)&(mass[:,None]>0)),tmp/'queries.pt')
        del field,data,teacher
        with (out/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-m','scripts.research.validate_query_field_compositional_atoms','worker','--root',str(tmp)],stdout=log,stderr=subprocess.STDOUT,check=True)
        for name in ['replay.pt','audit.json']:shutil.copyfile(tmp/name,out/name)
    expected=torch.load(p/'predictions.pt',weights_only=True)['scores'];actual=torch.load(out/'replay.pt',weights_only=True)
    errors={a:float((v-expected[a]).abs().max()) for a,v in actual.items()}
    exact={a:torch.equal(v>.5,expected[a]>.5) for a,v in actual.items()};audit=read(out/'audit.json')
    passed=max(errors.values())<=1e-6 and all(exact.values()) and audit['probe_blocked'] and not audit['forbidden_attempts']
    report=dict(passed=passed,scene=j['scene'],queries=actual['atom_parts'].shape[1],max_errors=errors,selected_exact=exact,
        source_denial=audit,source_assisted_query_fixtures=True,public_api_integrated=False,
        scope='One scene, all cached privileged acceptances; Python audit not OS sandbox. Decoder isolation only, not learned acceptance or deployment.',
        code=receipt(__file__),design=receipt(ROOT/'design.json'))
    write(out/'report.json',report);write('docs/experiments/2026-10-10-compositional-atoms-runtime.json',report);print(report,flush=True)
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','worker']);p.add_argument('--root');a=p.parse_args();torch.set_num_threads(4)
    worker(a.root) if a.stage=='worker' else run()
