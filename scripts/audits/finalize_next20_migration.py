"""Bounded registration, cold legacy-parent replay and standalone export; no deletion."""
import os,sys,json,time,uuid,fcntl,hashlib,subprocess
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,"/root/RADIO-GS-query-field")
from scripts.audits import execute_next20_migration as run
from radio_gs.query_field.storage import sha256
P=run.P;ROOT=run.ROOT;STORE=run.STORE
REV="4a0bd077663da181de01ddde8eadcbe559636296bec4ad0d93e9b8f2761ed100"
OUTPUT=Path("/mnt/pool/sqy/results/RADIO-GS/output")
DEPLOY=ROOT/"standalone_export_waldo_decoder_rollback"
EXECUTION_SHA=sha256(__file__)
def check_sources():
    if sha256(__file__)!=EXECUTION_SHA:raise ValueError("finalizer source changed during execution")
    snapshot=json.loads((ROOT/"source_snapshot.json").read_text())
    run.verify_execution(snapshot,REV)
    for e,old in zip(P["fields"],snapshot["fields"]):
        path=Path(e["source_directory"])
        assert run.fp(path/"field.pt")==old["fingerprint"] and sha256(path/"manifest.json")==old["manifest_sha256"]
        m=json.loads((Path(e["target_directory"])/"manifest.json").read_text())
        receipt=json.loads((Path(e["target_directory"])/"migration_verified.json").read_text())
        assert m["state_identity"]==receipt["state_identity"] and sha256(Path(e["target_directory"])/"manifest.json")==receipt["target_manifest_sha256"]
    return snapshot
def register():
    from radio_gs.query_field.legacy_field_resolver import create_mapping,FieldResolver
    target=Path(P["mapping_destination"])
    if target.exists():raise FileExistsError("registration destination already exists; inspect retained receipt before explicit retry")
    value=create_mapping([(e["source_directory"],e["target_directory"],str(STORE)) for e in P["fields"]],target,output_root=OUTPUT)
    resolver=FieldResolver(target,expected_mapping_sha256=sha256(target))
    for e in P["fields"]:resolver.inspect(e["source_directory"],e["source_field_sha256"])
    snap=json.loads((ROOT/"source_snapshot.json").read_text())
    read_bytes=2*sum(e["source_logical_bytes"] for e in P["fields"])+sum(e["tensor_bytes"] for e in snap["fields"])
    return dict(mapping=str(target),mapping_sha256=sha256(target),registered_fields=len(value["entries"]),application_payload_read_bytes=read_bytes,application_payload_written_bytes=0)
def registered_mapping():
    receipts=[json.loads(p.read_text()) for p in ROOT.glob("register_*.json")]
    receipts=[r for r in receipts if r.get("phase")=="register" and r.get("status")=="validated"]
    if len(receipts)!=1:raise ValueError("require one explicit validated registration receipt")
    receipt=receipts[0];path=Path(P["mapping_destination"])
    if receipt["mapping"]!=str(path) or sha256(path)!=receipt["mapping_sha256"]:raise ValueError("registered mapping identity changed")
    mapping=json.loads(path.read_text())
    wanted={(str(Path(e["source_directory"])/"field.pt"),e["source_field_sha256"],e["target_directory"],str(STORE)) for e in P["fields"]}
    actual={(e["old_path"],e["old_field_sha256"],e["target"],e["store_root"]) for e in mapping["entries"]}
    if len(mapping["entries"])!=20 or actual!=wanted:raise ValueError("mapping differs from exact approved 20 bindings")
    return path,receipt["mapping_sha256"]
def replay():
    from scripts.research import run_query_field_protected_v2 as runner
    mapping,mapping_sha=registered_mapping();replay_root=Path(P["versioned_replay_root"])
    if not replay_root.exists():
        runner.prepare(P["source_root"],sha256(Path(P["source_root"])/"design.json"),mapping,mapping_sha,replay_root,STORE)
    keys=P["stages"][6]["selected_scene_keys"]
    result=runner.build(replay_root,store_root=STORE,keys=keys)
    assert len(result["records"])==30 and sum(r["resolution"]["mode"]=="mapped_v2" for r in result["records"])==20
    snap=json.loads((ROOT/"source_snapshot.json").read_text())
    read_bytes=sum(e["tensor_bytes"] for e in snap["fields"])
    for key in keys:read_bytes+=3*(Path(P["source_root"])/"fits/functional_native"/key/"field/field.pt").stat().st_size
    return dict(resolved_construction=str(replay_root/"resolved_construction.json"),resolved_construction_sha256=sha256(replay_root/"resolved_construction.json"),resolved_arms=30,mapped_arms=20,unmigrated_v1_arms=10,training_launched=False,application_payload_read_bytes=read_bytes,application_payload_written_bytes=0)
def export():
    from radio_gs.query_field.legacy_field_resolver import FieldResolver
    e=P["fields"][0];mapping,mapping_sha=registered_mapping()
    resolver=FieldResolver(mapping,expected_mapping_sha256=mapping_sha)
    result=resolver.export_legacy(e["source_directory"],e["source_field_sha256"],DEPLOY,max_export_bytes=P["max_export_bytes"])
    child=subprocess.run([sys.executable,__file__,"cold"],capture_output=True,text=True,timeout=300)
    if child.returncode:raise RuntimeError("independent cold process failed: "+child.stderr[-2000:])
    cold=json.loads(child.stdout.strip().splitlines()[-1])
    n=(DEPLOY/"field.pt").stat().st_size
    tensor=json.loads((Path(e["target_directory"])/"manifest.json").read_text())["tensor_bytes"]
    return dict(standalone=result,cold=cold,export_allocated_bytes=(DEPLOY/"field.pt").stat().st_blocks*512,application_payload_read_bytes=tensor+4*n,application_payload_written_bytes=n)
def cold():
    import torch,builtins,io
    torch.set_num_threads(2)
    from radio_gs.query_field.storage import load_field
    from radio_gs.query_field.shared_storage import _descriptor
    from radio_gs.query_field import QueryPacket,Support
    expected=json.loads((Path(P["fields"][0]["target_directory"])/"manifest.json").read_text())
    original_open=builtins.open;original_io_open=io.open
    def guard(function):
        def opened(path,*args,**kwargs):
            if not isinstance(path,int) and os.path.commonpath([str(DEPLOY),os.path.abspath(path)])!=str(DEPLOY):
                raise AssertionError("undeclared cold deployment dependency: "+str(path))
            return function(path,*args,**kwargs)
        return opened
    builtins.open=guard(original_open);io.open=guard(original_io_open)
    field,loaded_manifest=load_field(DEPLOY)
    validated_archive_sha=loaded_manifest["files"]["field.pt"]["sha256"]
    assert validated_archive_sha==P["fields"][0]["source_field_sha256"]
    assert {k:_descriptor(v)[0] for k,v in field.state_dict().items()}==expected["tensors"]
    assert field.export_spec()==expected["spec"]
    row=int(torch.where(field.observed)[0][0])
    packet=QueryPacket("point_3d",positive=Support(torch.tensor([row]),torch.ones(1)),exact_positive=True)
    result=field.query(packet)
    assert torch.isfinite(result.membership_scores).all()
    print(json.dumps(dict(cold_process=True,python_open_guard_during_load_and_query=True,all_state_and_spec_exact=True,point_query_finite=True,archive_sha256=validated_archive_sha)))
def main(mode):
    run.validate_plan_paths()
    with (ROOT/".execution.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_sources();before=run.resources()
        run.visible_refs([Path(e["source_directory"])/"field.pt" for e in P["fields"]])
        reservation={"register":160000000000,"replay":200000000000,"export":50000000000}[mode]
        if run.charged_io()+reservation>P["conservative_total_io_budget_bytes"]:raise RuntimeError("remaining cumulative I/O budget insufficient")
        attempt=mode+"_"+uuid.uuid4().hex;index={"register":106,"replay":107,"export":108}[mode]
        run.event(dict(kind="field_reserved",attempt=attempt,index=index,reserved_application_bytes=reservation))
        start=time.monotonic()
        try:
            result={"register":register,"replay":replay,"export":export}[mode]()
            check_sources()
            if mode in {"replay","export"}:registered_mapping()
            report=dict(phase=mode,attempt=attempt,index=index,status="validated",elapsed_seconds=time.monotonic()-start,resources_before=before,resources_after=run.resources(),execution_script_sha256=EXECUTION_SHA,originals_preserved=True,**result)
            assert report["application_payload_read_bytes"]+report["application_payload_written_bytes"]<=reservation
            run.write(ROOT/(attempt+".json"),report)
            run.event(dict(kind="field_validated",**report))
            print(json.dumps(report),flush=True)
        except Exception as exc:
            run.event(dict(kind="phase_failed",attempt=attempt,phase=mode,error=repr(exc)))
            raise
if __name__=="__main__":
    if sys.argv[1]=="cold":cold()
    else:main(sys.argv[1])
