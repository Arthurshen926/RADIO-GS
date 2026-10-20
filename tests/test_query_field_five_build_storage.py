import json
from pathlib import Path

import pytest

from scripts.research import run_query_field_five_build as runner
from radio_gs.query_field.shared_storage import save_field_v2
from radio_gs.query_field.storage import save_field
from test_query_preserving_field import example


def test_five_build_requires_explicit_storage_before_cohort_read():
    with pytest.raises(SystemExit) as error:
        runner.main(["--cohort","/nonexistent/cohort.json"])
    assert error.value.code == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_mock_five_build_propagates_storage_and_costs(tmp_path, monkeypatch, legacy):
    jobs = []
    for i in range(5):
        root=tmp_path/f"job{i}";root.mkdir()
        config=root/"config.json";config.write_text("{}")
        jobs.append(dict(name=f"scene{i}",family=["nvos","scannet","uqis"][i%3],
                         output=str(root),config=str(config),needs_sam=False,
                         train_frames=[1],development_frames=[2]))
    cohort=tmp_path/"cohort.json";cohort.write_text(json.dumps({"construction":jobs}))
    commands=[]
    def execute(command, log, environment):
        commands.append(command)
        stage=command[3];root=Path(command[command.index("--output")+1])
        if stage=="compile":
            (root/"observations").mkdir()
            (root/"observations/observations.json").write_text(json.dumps(
                {"geometry_condition":"synthetic","observed_elements":12}))
        else:
            assert stage=="fit"
            if legacy:
                assert "--legacy-v1" in command and "--store-root" not in command
                save_field(example(),root/"models/trained",{})
            else:
                assert "--legacy-v1" not in command
                store=command[command.index("--store-root")+1]
                save_field_v2(example(),root/"models/trained",{},store_root=store,reserve_bytes=0)
    monkeypatch.setattr(runner,"execute",execute)
    args=["--cohort",str(cohort),"--summary-root",str(tmp_path/"summary"),"--gpus","0"]
    args += ["--legacy-v1"] if legacy else ["--store-root",str(tmp_path/"store")]
    runner.main(args)
    result=json.loads((tmp_path/"summary/construction_complete.json").read_text())
    assert len(commands)==10 and len(result["jobs"])==5
    assert result["storage_mode"] == ("explicit_legacy_v1" if legacy else "shared_v2")
    for row in result["jobs"]:
        assert row["field_schema"].endswith(".v1" if legacy else ".v2")
        assert row["deployment_bytes"]>0 and row["observed"]==12

@pytest.mark.parametrize('case', ['legacy_with_v2', 'different_store', 'missing_store'])
def test_cached_build_rejects_mode_or_store_mismatch(tmp_path, case):
    root=tmp_path/'job';root.mkdir()
    store=tmp_path/'store'
    manifest=save_field_v2(example(),root/'models/trained',{},store_root=store,reserve_bytes=0)
    (root/'build_complete.json').write_text(json.dumps(dict(
        blob_store=str(store),field_schema=manifest['schema'],field_sha256=manifest['state_identity'])))
    job={'output':str(root)}
    if case=='legacy_with_v2':
        with pytest.raises(ValueError,match='format'):
            runner.build(job,{},0,legacy_v1=True)
    else:
        other=tmp_path/'other'
        if case=='different_store': other.mkdir()
        with pytest.raises(ValueError,match='store|root|closure'):
            runner.build(job,{},0,store_root=other)
    assert runner.build(job,{},0,store_root=store)['blob_store']==str(store)

@pytest.mark.parametrize('missing', [True, False])
def test_partial_resume_rejects_missing_or_empty_store(tmp_path, missing):
    root=tmp_path/'job';root.mkdir()
    save_field_v2(example(),root/'models/trained',{},store_root=tmp_path/'real-store',reserve_bytes=0)
    (root/'observations').mkdir()
    (root/'observations/observations.json').write_text('{}')
    other=tmp_path/'wrong-store'
    if not missing: other.mkdir()
    with pytest.raises(ValueError,match='root|closure'):
        runner.build({'output':str(root),'needs_sam':False},{},0,store_root=other)
    assert not (root/'build_complete.json').exists()


