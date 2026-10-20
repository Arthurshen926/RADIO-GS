import json
from pathlib import Path

import pytest

from scripts.research.run_shared_evidence_migration import (
    available_gpus, check_traces, prediction_seals, validate_recipe)


def test_recipe_gate_rejects_missing_and_duplicate_tasks():
    config = dict(cohorts=dict(lerf=['room1','room2'],nvos=['reference']))
    jobs = [dict(benchmark='lerf',name='room1'),dict(benchmark='lerf',name='room2'),
            dict(benchmark='nvos',name='reference')]
    validate_recipe(dict(jobs=jobs),config)
    with pytest.raises(ValueError,match='cohort'):
        validate_recipe(dict(jobs=jobs[:-1]),config)
    with pytest.raises(ValueError,match='duplicate'):
        validate_recipe(dict(jobs=jobs+[jobs[0]]),config)


def test_call_trace_requires_real_admission_and_member_operations(tmp_path):
    jobs = [dict(benchmark=b,name='room') for b in ['lerf','nvos','scannet','uqis']]
    for job in jobs:
        p=tmp_path/job['benchmark']/job['name']/'core_trace';p.mkdir(parents=True)
        (p/'core_1.json').write_text(json.dumps(dict(calls=dict(project=3))))
    with pytest.raises(ValueError,match='not exercised'):
        check_traces(tmp_path,dict(jobs=jobs))
    for job in jobs:
        p=tmp_path/job['benchmark']/job['name']/'core_trace/core_1.json'
        p.write_text(json.dumps(dict(calls=dict(project=3,compile_domains=2,
                                               grouped_membership_features=1,scope_response=1))))
    assert len(check_traces(tmp_path,dict(jobs=jobs)))==4


def test_prediction_seal_rejects_missing_outputs_and_binds_bytes(tmp_path):
    recipe=dict(jobs=[dict(benchmark='nvos',name='room')])
    with pytest.raises(FileNotFoundError):prediction_seals(tmp_path,recipe)
    path=tmp_path/'nvos/room/query/typed_readout.pt';path.parent.mkdir(parents=True)
    path.write_bytes(b'actual output')
    old=prediction_seals(tmp_path,recipe)
    path.write_bytes(b'changed output')
    assert old!=prediction_seals(tmp_path,recipe)


def test_gpu_guard_requires_both_headroom_and_idle_device(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0,
        stdout='0, 18000, 100\n1, 1000, 0\n2, 18000, 0\n',stderr=''))
    ready,_=available_gpus(16384)
    assert ready==['2']


def test_preflight_seals_auxiliary_scene_and_provenance_without_maps(tmp_path,monkeypatch):
    from scripts.research import run_shared_evidence_migration as runner
    import hashlib
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    sidecar=tmp_path/'provenance.json';sidecar.write_text('{}')
    auxiliary=tmp_path/'aux.json';auxiliary.write_text(json.dumps(dict(
        native_observation_field='not-opened.pt',native_observation_provenance=dict(
            path=str(sidecar),sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest()))))
    scene=tmp_path/'scene.json';scene.write_text(json.dumps(dict(local_auxiliary_scene='aux.json')))
    assert runner.scene_dependencies(scene)=={scene,auxiliary,sidecar}
    sidecar.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash changed'):runner.scene_dependencies(scene)


@pytest.mark.parametrize('over_budget',[False,True])
def test_runner_preserves_reports_and_releases_only_owned_transients(tmp_path,monkeypatch,over_budget):
    from scripts.research import run_shared_evidence_migration as runner
    recipe=tmp_path/'recipe.json'
    recipe.write_text(json.dumps(dict(jobs=[dict(benchmark='nvos',name='room')])))
    ram=tmp_path/'ram';ram.mkdir();sentinel=ram/'shared.bin';sentinel.write_bytes(b'keep shared input')
    configuration=tmp_path/'migration.json'
    configuration.write_text(json.dumps(dict(recipe='recipe.json',study_root='study',storage=dict(
        transient_root=str(ram),transient_bytes=100 if over_budget else 2**20,transient_reserve_bytes=1,
        durable_policy=dict(study_bytes=4*2**20,run_bytes=2*2**20,artifact_bytes=2**20,reserve_bytes=1)))))
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'CONFIG',configuration)
    monkeypatch.setattr(runner,'manifest',lambda config,arm,**kwargs:dict(ready_gpus=['0']))
    monkeypatch.setattr('sys.argv',['runner','--run','fixture','--execute'])
    monkeypatch.chdir(tmp_path)

    class CompletedCohort:
        returncode=0
        def __init__(self,command,**kwargs):
            root=Path(command[command.index('--output')+1])
            query=root/'nvos/room/query';query.mkdir(parents=True)
            (query/'typed_readout.pt').write_bytes(b'new fixture prediction')
            trace=query.parent/'core_trace';trace.mkdir()
            (trace/'core_1.json').write_text(json.dumps(dict(calls=dict(compile_domains=2,project=3))))
            (root/'summary.json').write_text(json.dumps(dict(completed=1,macro=dict(nvos=dict(iou=.8)),scannet_paper8={})))
            (root/'prediction_seal.json').write_text(json.dumps(dict(
                outputs=prediction_seals(root,dict(jobs=[dict(benchmark='nvos',name='room')])),sealed_before_scoring=True)))
            (root/'query.log').write_text('full inference log\n'*10)
        def poll(self):return 0

    monkeypatch.setattr('subprocess.Popen',CompletedCohort)
    if over_budget:
        with pytest.raises(RuntimeError,match='budget exceeded'):runner.main()
    else:
        runner.main()
        assert (tmp_path/'study/fixture/reports/summary.json').is_file()
        assert (tmp_path/'study/fixture/prediction_hashes.json').is_file()
        import gzip
        assert gzip.decompress((tmp_path/'study/fixture/logs/cohort/query.log.gz').read_bytes()).decode()=='full inference log\n'*10
    outcome=json.loads((tmp_path/'study/fixture/outcome.json').read_text())
    assert outcome['status']==('failed' if over_budget else 'complete')
    assert not list((tmp_path/'study').rglob('*.pt'))
    assert list(ram.iterdir())==[sentinel]
    assert sentinel.read_bytes()==b'keep shared input'
