"""Frozen-profile, storage-bounded fresh five-output regression.

Default: prepare a reviewable preflight manifest only. --execute requires a free
GPU, writes dense outputs to a private RAM directory, and retains small reports
and prediction hashes. It never reuses historical predictions or deletes shared
assets. Source teacher caches remain the declared inputs of the parity profile.
"""
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from radio_gs.pipeline.prediction_receipts import prediction_seals

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/research/shared_evidence_migration.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def validate_recipe(recipe, config):
    jobs = recipe['jobs']; pairs = [(j['benchmark'], j['name']) for j in jobs]
    if len(pairs) != len(set(pairs)):
        raise ValueError('duplicate declared scene/task')
    for benchmark, names in config['cohorts'].items():
        if {j['name'] for j in jobs if j['benchmark'] == benchmark} != set(names):
            raise ValueError('incomplete or changed cohort: '+benchmark)
    if {j['benchmark'] for j in jobs} != set(config['cohorts']):
        raise ValueError('unexpected benchmark in migration recipe')


def available_gpus(minimum_mib):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode:
        return [], dict(error=result.stderr.strip())
    rows = [dict(zip(['index','free_mib','utilization'],map(int,line.split(','))))
            for line in result.stdout.splitlines() if line.strip()]
    # This is a launch guard, not an exclusive GPU reservation.
    return [str(r['index']) for r in rows if r['free_mib'] >= minimum_mib and r['utilization'] <= 10], rows


def scene_dependencies(scene_path, seen=None):
    """Bind both teachers and explicit metadata overlays, without opening RGB/maps."""
    seen=set() if seen is None else seen
    path=Path(scene_path).resolve()
    if path in seen:return set()
    seen.add(path);paths={path};scene=read(path)
    if scene.get('local_auxiliary_scene'):
        paths.update(scene_dependencies(ROOT/scene['local_auxiliary_scene'],seen))
    reference=scene.get('native_observation_provenance')
    if reference:
        overlay=ROOT/reference['path']
        if sha(overlay)!=reference['sha256']:
            raise ValueError('region provenance hash changed before launch')
        paths.add(overlay.resolve())
    return paths


def validate_extension_recipe(recipe, base):
    """Only the declared reference extension may differ from the CPU baseline."""
    old={(j['benchmark'],j['name']):j for j in base['jobs']}
    if len(recipe['jobs'])!=len(old):raise ValueError('extension cohort differs')
    policies=set()
    for job in recipe['jobs']:
        key=(job['benchmark'],job['name'])
        if key not in old:raise ValueError('extension cohort differs')
        if job['benchmark']=='nvos':
            if 'reference_completion' not in job:raise ValueError('partial N14 extension')
            profile=read(ROOT/job['reference_completion']);policies.add(profile['policy'])
            stripped={k:v for k,v in job.items() if k not in ('reference_completion','prompt_contract','input_track')}
            expected={k:v for k,v in old[key].items() if k not in ('prompt_contract','input_track')}
            if stripped!=expected:raise ValueError('N14 extension changed initializer recipe')
        elif job!=old[key]:raise ValueError('N14 extension changed other benchmarks')
    if len(policies)!=1:raise ValueError('task-wise reference policy selection is forbidden')


def manifest(config, arm, config_path=None):
    recipe = read(ROOT/config['recipe']); validate_recipe(recipe, config)
    paths = {Path(config_path) if config_path else CONFIG, ROOT/config['recipe'], Path(__file__).resolve(),
             ROOT/config['protocol_freeze'], ROOT/config['component_audit']}
    for directory in ['radio_gs/evidence_core','radio_gs/querying','radio_gs/pipeline']:
        paths.update((ROOT/directory).glob('*.py'))
    paths.update(ROOT/p for p in config['evaluators'])
    for job in recipe['jobs']:
        paths.update(scene_dependencies(ROOT/job['scene']))
        if job.get('reference_completion'):paths.add(ROOT/job['reference_completion'])
        for key in ['queries','evaluation']:
            if key in job:
                p = ROOT/job[key]
                if p.is_file():paths.add(p)
                elif p.is_dir():paths.update((p/'queries').glob('*.json'))
    audit = read(ROOT/config['component_audit'])
    component_recipe=config.get('base_component_recipe',config['recipe'])
    if (Path(audit['recipe']).resolve() != (ROOT/component_recipe).resolve()
            or audit.get('recipe_sha256') != sha(ROOT/component_recipe)):
        raise ValueError('component audit does not bind this recipe')
    if 'base_component_recipe' in config:
        validate_extension_recipe(recipe,read(ROOT/component_recipe))
        extension_path=ROOT/config['extension_cpu_audit'];extension=read(extension_path);paths.add(extension_path)
        if not extension['passed'] or extension['gpu_end_to_end_validated']:
            raise ValueError('invalid extension CPU gate')
        for p,h in {**extension['source_sha256'],**extension['profiles_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('extension CPU audit is stale: '+p)
        if extension['profiles_sha256'].get(config['recipe'])!=sha(ROOT/config['recipe']):
            raise ValueError('extension audit does not bind the selected recipe')
    if (audit['passed'] != len(recipe['jobs']) or audit['fresh_benchmark_evaluation'] or
            not audit.get('sources_unchanged_during_audit')):
        raise ValueError('component migration gate is incomplete or mislabeled')
    if not audit.get('source_sha256'):
        raise ValueError('component audit lacks source bindings')
    for path,digest in audit['source_sha256'].items():
        if sha(ROOT/path)!=digest:
            raise ValueError('component audit is stale: '+path)
    hashes = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):sha(p) for p in sorted(paths)}
    gpus, devices = available_gpus(config['minimum_free_gpu_mib'])
    return dict(schema='radiogs.shared_core_migration_run.v1',arm=arm,recipe=config['recipe'],
        outputs=config['outputs'],cohorts=config['cohorts'],protocols=config['protocols'],
        code_and_configuration_sha256=hashes,component_audit_passed=True,
        asset_binding='existing declarations plus sampled audit hashes; not an exhaustive cache-content snapshot',
        ready_gpus=gpus,gpu_preflight=devices,storage=config['storage'],
        inference_started=False,fresh_benchmark_evaluation=False,
        historical_scores_are_new_results=False,default_promotion=False,
        pending_extensions=config['pending_extensions'],
        extension_cpu_audit=config.get('extension_cpu_audit'),gpu_end_to_end_validated=False)


def check_traces(root, recipe):
    required={'lerf':['compile_domains','project'],'nvos':['compile_domains','project'],
              'scannet':['compile_domains','project'],
              'uqis':['grouped_membership_features','scope_response','project']}
    rows={}
    for job in recipe['jobs']:
        counts={}
        for path in (root/job['benchmark']/job['name']/'core_trace').glob('*.json'):
            for operation,count in read(path)['calls'].items():
                counts[operation]=counts.get(operation,0)+count
        if any(counts.get(key,0)==0 for key in required[job['benchmark']]):
            raise ValueError('shared core not exercised by '+job['benchmark']+'/'+job['name'])
        if job.get('reference_completion') and not counts.get('reference_completion_native_writeback'):
            raise ValueError('reference completion writeback not exercised by '+job['name'])
        rows[job['benchmark']+'/'+job['name']]=counts
    return rows


def retain_logs(run, temporary):
    """Lossless log retention. Exceeding a budget fails; no silent truncation."""
    for path in sorted(Path(temporary).rglob('*.log')):
        data=gzip.compress(path.read_bytes(),mtime=0)
        relative='logs/'+str(path.relative_to(temporary))+'.gz'
        run._write(relative,lambda f,value=data:f.write(value),limit=len(data),nested=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['legacy','shared_v1'],default='shared_v1')
    parser.add_argument('--run',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args();os.chdir(ROOT)
    config=read(args.config); state=manifest(config,args.arm,config_path=args.config.resolve())
    storage=config['storage'];run=BoundedExperiment(ROOT/config['study_root'],args.run,storage['durable_policy'])
    run.json('manifest.json',state)
    if not args.execute:
        print(json.dumps(dict(manifest=str(run.root/'manifest.json'),ready_gpus=state['ready_gpus'],
            inference_started=False),indent=2));return
    if not state['ready_gpus']:
        run.json('outcome.json',dict(status='not_started_gpu_busy',fresh_benchmark_evaluation=False))
        raise SystemExit('No idle GPU satisfies the declared memory guard; no inference started.')
    ram=Path(storage['transient_root']);limit=storage['transient_bytes'];reserve=storage['transient_reserve_bytes']
    if shutil.disk_usage(ram).free < limit+reserve:
        raise RuntimeError('insufficient RAM filesystem capacity for the declared transient budget')
    recipe=read(config['recipe']);gpu=state['ready_gpus'][0];started=time.monotonic()
    outcome=dict(status='failed',fresh_benchmark_evaluation=False)
    # Only this newly created private directory is removed. Historical/shared
    # outputs are read-only inputs throughout this runner.
    with tempfile.TemporaryDirectory(prefix='radiogs-core-',dir=ram) as tmp:
        root=Path(tmp)/'cohort';log=Path(tmp)/'cohort.log'
        command=[sys.executable,'-m','radio_gs.pipeline.cohort','--recipe',config['recipe'],
                 '--output',str(root),'--gpus',gpu,'--evidence-core',args.arm,'--scannet-paper8']
        run.json('launch.json',dict(command=command,gpu=gpu,started=time.time()))
        process=None;peak=0
        try:
            with log.open('w') as handle:
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
                while process.poll() is None:
                    peak=max(peak,tree_bytes(Path(tmp)))
                    if peak>limit or shutil.disk_usage(ram).free<reserve:
                        raise RuntimeError('transient storage budget exceeded')
                    time.sleep(5)
            peak=max(peak,tree_bytes(Path(tmp)))
            if peak>limit or shutil.disk_usage(ram).free<reserve:
                raise RuntimeError('transient storage budget exceeded')
            if process.returncode:raise RuntimeError('cohort subprocess exit '+str(process.returncode))
            if args.arm=='shared_v1':run.json('core_trace.json',check_traces(root,recipe))
            sealed=read(root/'prediction_seal.json')
            if not sealed['sealed_before_scoring'] or sealed['outputs']!=prediction_seals(root,recipe):
                raise ValueError('predictions changed after the pre-evaluation seal')
            run.json('prediction_hashes.json',sealed)
            report_names={'summary.json','result.json','evaluation.json','report.json','fit.json',
                          'execution.json','predictions_complete.json','prediction_seal.json'}
            for p in sorted(root.rglob('*.json')):
                if p.name in report_names or p.parent.name=='runs':
                    run.public_artifact(run.root/'reports'/p.relative_to(root),read(p),'json')
            retain_logs(run,tmp)
            summary=read(root/'summary.json')
            if summary['completed']!=len(recipe['jobs']):raise ValueError('partial cohort cannot close migration')
            outcome=dict(status='complete',fresh_benchmark_evaluation=True,macro=summary['macro'],
                         scannet_paper8=summary['scannet_paper8'],default_promotion=False)
        except BaseException as error:
            outcome['error']=repr(error)
            raise
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            run.json('log_tail.json',dict(text=log.read_text(errors='replace')[-32768:] if log.exists() else ''))
            run.json('outcome.json',dict(outcome,seconds=time.monotonic()-started,peak_transient_bytes=peak,
                durable_dense_prediction_bytes=0,transient_predictions_released=True))
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':main()
