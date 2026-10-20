import json
from pathlib import Path

import pytest

from test_query_preserving_field import example
from radio_gs.query_field.shared_storage import save_field_v2
from radio_gs.query_field.replay_v2 import prepare_replay


def test_replay_keeps_frozen_config_and_other_values(tmp_path):
    save_field_v2(example(), tmp_path/"v2", {"source_field_sha256":"a"*64}, reserve_bytes=0)
    config = {"jobs":[{"name":"a","field":"old/a","output":"old/output"},{"name":"b","field":"old/b"}]}
    source = tmp_path/"frozen.json"
    source.write_text(json.dumps(config))
    before = source.read_bytes()
    receipt = prepare_replay(tmp_path/"v2", None, source, "/jobs/0/field", "old/a",
                             tmp_path/"standalone", tmp_path/"replay.json", expected_source_sha256="a"*64)
    after = json.loads((tmp_path/"replay.json").read_text())
    assert source.read_bytes() == before
    assert after["jobs"][0]["field"] == str(tmp_path/"standalone")
    after["jobs"][0]["field"] = "old/a"
    assert after == config
    assert receipt["cold_query_file_access_denied"] and not receipt["benchmark_launched"]
    with pytest.raises(FileExistsError):
        prepare_replay(tmp_path/"v2", None, source, "/jobs/0/field", "old/a",
                       tmp_path/"standalone", tmp_path/"replay.json", expected_source_sha256="a"*64, reuse_deployment=True)


def test_replay_wrong_pointer_fails_before_export(tmp_path):
    source=tmp_path/"config.json"
    source.write_text('{"field":"old/a"}')
    with pytest.raises(ValueError, match="explicitly selected"):
        prepare_replay(tmp_path/"missing", None, source, "/field", "wrong",
                       tmp_path/"standalone", tmp_path/"replay.json")
    assert not (tmp_path/"standalone").exists()


def test_replay_source_digest_and_slot_budget_fail_closed(tmp_path):
    save_field_v2(example(), tmp_path/"v2", {"source_field_sha256":"a"*64}, reserve_bytes=0)
    source=tmp_path/"config.json";source.write_text('{"field":"old/a"}')
    for digest, budget, message in [("b"*64, 1024**3, "legacy source digest"), ("a"*64, 0, "slot budget")]:
        with pytest.raises(ValueError, match=message):
            prepare_replay(tmp_path/"v2", None, source, "/field", "old/a",
                           tmp_path/"slot", tmp_path/"new.json",
                           expected_source_sha256=digest, max_export_bytes=budget)
        assert not (tmp_path/"slot").exists() and not (tmp_path/"new.json").exists()


def test_pipeline_migration_binds_source_for_replay_and_rejects_lossy_legacy(tmp_path):
    from radio_gs.query_field.pipeline_v2 import main
    from radio_gs.query_field.storage import save_field
    original=save_field(example(),tmp_path/"old",{})
    main(["migrate-v1","--field",str(tmp_path/"old"),"--output",str(tmp_path/"v2"),
          "--store-root",str(tmp_path/"store"),"--reserve-bytes","0"])
    manifest=json.loads((tmp_path/"v2/manifest.json").read_text())
    assert manifest["provenance"]["source_field_sha256"] == original["files"]["field.pt"]["sha256"]
    f=example().double();f.xyz[0,0]=1.0000000001
    save_field(f,tmp_path/"double",{})
    with pytest.raises(ValueError,match="legacy constructor"):
        main(["migrate-v1","--field",str(tmp_path/"double"),"--output",str(tmp_path/"invalid"),
              "--store-root",str(tmp_path/"store"),"--reserve-bytes","0"])
    assert not (tmp_path/"invalid").exists()
