"""Bounded next20 conversion; no source mutation or deletion."""
import os,sys,json,time,shutil,subprocess,gc,stat,hashlib,fcntl
if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):raise RuntimeError("optimized Python is forbidden for this audited script")
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0,"/root/RADIO-GS-query-field")
PLAN=Path("/root/RADIO-GS-query-field/docs/query-field-next20-migration-plan.json")
PLAN_BYTES=PLAN.read_bytes();PLAN_SHA=hashlib.sha256(PLAN_BYTES).hexdigest()
P=json.loads(PLAN_BYTES); ROOT=Path(P["target_root"]); STORE=Path(P["store_root"])
def validate_plan_paths():
    assert P["field_count"]==len(P["fields"])==20 and [e["index"] for e in P["fields"]]==list(range(20))
    assert [i for s in P["stages"][:5] for i in s["field_indices"]]==list(range(20))
    for e in P["fields"]:
        Path(e["source_directory"]).relative_to(P["source_root"])
        Path(e["target_directory"]).relative_to(ROOT)
        assert os.path.commonpath([str(ROOT),e["source_directory"]])!=str(ROOT)
    for path in [ROOT,STORE]+[Path(e[k]) for e in P["fields"] for k in ["source_directory","target_directory"]]:
        assert path.is_absolute()
        for part in [path]+list(path.parents):
            if part.is_symlink():raise ValueError("symlink path forbidden: "+str(part))
def write(path,value):
    with path.open("x") as h:json.dump(value,h,indent=2);h.write("\n");h.flush();os.fsync(h.fileno())
def event(value):
    with (ROOT/"migration_journal.jsonl").open("a") as h:h.write(json.dumps(dict(time=time.time(),**value))+"\n");h.flush();os.fsync(h.fileno())
def fp(path):
    s=path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError("source not regular nlink1: "+str(path))
    return {k:getattr(s,k) for k in ["st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_nlink","st_blocks"]}
def resources():
    ram=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024
    free=shutil.disk_usage(STORE).free
    if free<P["required_free_bytes"] or ram<P["minimum_available_ram_bytes"]:raise RuntimeError("free/RAM gate failed")
    return dict(df_free_bytes=free,available_ram_bytes=ram)
def visible_refs(paths):
    ids={(p.stat().st_dev,p.stat().st_ino) for p in paths};refs=[];unreadable=[];jobs=[]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
            if any(x in cmd.lower() for x in ["radio-gs","radiogs","anygs"]) and "execute_next20_migration.py" not in cmd and "python -c" not in cmd:jobs.append([proc.name,cmd[:300]])
            for fd in (proc/"fd").iterdir():
                try:s=fd.stat()
                except FileNotFoundError:continue
                if (s.st_dev,s.st_ino) in ids:refs.append([proc.name,str(fd)])
            for line in (proc/"maps").read_text().splitlines():
                parts=line.split()
                if len(parts)>4:
                    dev=parts[3].split(":")
                    if (os.makedev(int(dev[0],16),int(dev[1],16)),int(parts[4])) in ids:refs.append([proc.name,"mmap"])
        except FileNotFoundError:pass
        except PermissionError:unreadable.append(proc.name)
    if refs or jobs:raise RuntimeError("visible active references/jobs: "+repr([refs,jobs]))
    return dict(visible_source_references=refs,visible_relevant_jobs=jobs,inaccessible_proc_pids=unreadable,other_hosts_not_visible=True)
def code_hashes():
    paths=[Path(__file__)]+list(Path("/root/RADIO-GS-query-field/radio_gs/query_field").glob("*.py"))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def snapshot():
    inventory=json.loads(Path("/root/RADIO-GS-query-field/docs/experiments/2026-10-12-next-native-unchanged-candidates.json").read_text())
    candidates={e["field_directory"]:e for e in inventory["candidates"]}
    from radio_gs.query_field.storage import sha256
    values=[]
    for e in P["fields"]:
        source=Path(e["source_directory"]);m=json.loads((source/"manifest.json").read_text());c=candidates[str(source)]
        assert m["schema"]=="radio_gs.query_preserving_field.v1" and m["external_scene_dependencies"]==[] and set(m["files"])=={"field.pt"}
        assert m["files"]["field.pt"]["sha256"]==e["source_field_sha256"] and m["files"]["field.pt"]["bytes"]==e["source_logical_bytes"]
        assert m["provenance"]==c["provenance"] and m["elements"]==c["elements"]
        f=fp(source/"field.pt");assert f["st_size"]==e["source_logical_bytes"]
        values.append(dict(index=e["index"],fingerprint=f,manifest_sha256=sha256(source/"manifest.json"),tensor_bytes=m["tensor_bytes"]))
    assert sha256(PLAN)==PLAN_SHA
    return dict(plan_sha256=PLAN_SHA,code_sha256=code_hashes(),fields=values,existing_store_blobs={p.stem:p.stat().st_size for p in STORE.glob("*.bin")},resources=resources())
def verify_execution(snapshot,revision_sha):
    assert snapshot["plan_sha256"]==PLAN_SHA==hashlib.sha256(PLAN.read_bytes()).hexdigest()
    current=code_hashes()
    if current==snapshot["code_sha256"]:
        assert not revision_sha
        return
    if not revision_sha:raise ValueError("changed execution code requires explicit revision SHA")
    path=ROOT/"execution_revision_v2.json"
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=revision_sha:raise ValueError("execution revision SHA differs")
    revision=json.loads(data)
    assert revision["original_snapshot_sha256"]==hashlib.sha256((ROOT/"source_snapshot.json").read_bytes()).hexdigest()
    assert revision["previous_code_sha256"]==snapshot["code_sha256"] and revision["new_code_sha256"]==current
    assert revision["stage1_summary_sha256"]==hashlib.sha256((ROOT/"stage1_summary.json").read_bytes()).hexdigest()
def charged_io():
    journal=ROOT/"migration_journal.jsonl"
    ledger={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            e=json.loads(line)
            if e["kind"]=="field_reserved":ledger[(e["attempt"],e["index"])]=e["reserved_application_bytes"]
            elif e["kind"]=="field_validated":ledger[(e["attempt"],e["index"])]=e["application_payload_read_bytes"]+e["application_payload_written_bytes"]
    return sum(ledger.values())
def one(index,attempt,revision_sha=""):
    import torch
    torch.set_num_threads(2)
    from radio_gs.query_field.storage import sha256,field_from_state
    from radio_gs.query_field.shared_storage import _descriptor,_safe_root,estimate_storage,save_field_v2,load_field_v2
    e=P["fields"][index]; source=Path(e["source_directory"]);target=Path(e["target_directory"]);old=source/"field.pt"
    _safe_root(source);_safe_root(STORE)
    snap=json.loads((ROOT/"source_snapshot.json").read_text());expected=snap["fields"][index]
    verify_execution(snap,revision_sha)
    assert fp(old)==expected["fingerprint"] and sha256(source/"manifest.json")==expected["manifest_sha256"]
    before=resources();refs=visible_refs([old]);start=time.monotonic()
    event(dict(kind="field_start",attempt=attempt,index=index,resources=before,process_references=refs))
    assert sha256(old)==e["source_field_sha256"]
    raw=torch.load(old,map_location="cpu",weights_only=True);m=json.loads((source/"manifest.json").read_text())
    field=field_from_state(raw,m,preserve_dtypes=True)
    desc={k:_descriptor(v)[0] for k,v in raw.items()}
    assert desc=={k:_descriptor(v)[0] for k,v in field.state_dict().items()}
    spec=field.export_spec();del raw;gc.collect()
    estimate=estimate_storage(field,STORE)
    existing_read=None
    resumed=(target/"manifest.json").is_file()
    if not resumed:
        assert not target.exists(),"partial destination retained; explicit recovery path required"
        save_field_v2(field,target,dict(source_manifest=m,source_field_sha256=e["source_field_sha256"]),store_root=STORE,max_new_bytes=8*1024**3,reserve_bytes=P["reserve_bytes"])
    del field;gc.collect()
    restored,new=load_field_v2(target,store_root=STORE)
    assert new["provenance"]==dict(source_manifest=m,source_field_sha256=e["source_field_sha256"])
    assert desc=={k:_descriptor(v)[0] for k,v in restored.state_dict().items()} and restored.export_spec()==spec
    assert fp(old)==expected["fingerprint"] and sha256(source/"manifest.json")==expected["manifest_sha256"]
    after=resources()
    result=dict(attempt=attempt,index=index,source=str(old),source_sha256=e["source_field_sha256"],source_fingerprint=expected["fingerprint"],source_manifest_sha256=expected["manifest_sha256"],target=str(target),target_manifest_sha256=sha256(target/"manifest.json"),state_identity=new["state_identity"],raw_state_dtype_shape_value_exact=True,spec_exact=True,original_preserved=True,resumed=resumed,new_blob_bytes_this_attempt=0 if resumed else new["new_blob_bytes"],referenced_unique_bytes=new["unique_blob_bytes"],elapsed_seconds=time.monotonic()-start,resources_before=before,resources_after=after,application_payload_read_bytes=2*e["source_logical_bytes"]+new["tensor_bytes"]+(0 if resumed else new["tensor_bytes"]-new["new_blob_bytes"]),application_payload_written_bytes=0 if resumed else new["new_blob_bytes"],io_accounting="known whole-payload operations, excludes metadata; not backend NFS measurement")
    receipt=target/"migration_verified.json"
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        assert all(previous[k]==result[k] for k in ["state_identity","source_sha256","source_manifest_sha256","target_manifest_sha256"])
    else:write(receipt,result)
    write(ROOT/"attempts"/attempt/("field%d.json"%index),result)
    event(dict(kind="field_validated",**result));print(json.dumps(result),flush=True)
def stage(number,revision_sha=""):
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (ROOT/"source_snapshot.json").exists():write(ROOT/"source_snapshot.json",snapshot())
    verify_execution(json.loads((ROOT/"source_snapshot.json").read_text()),revision_sha)
    st=P["stages"][number-1];assert st["kind"]=="migrate_existing_v1_to_v2"
    import uuid
    attempt="stage%d_"%number+uuid.uuid4().hex
    (ROOT/"attempts"/attempt).mkdir(parents=True,exist_ok=False)
    start=time.monotonic();before=resources();results=[]
    for index in st["field_indices"]:
        reserved=2*P["fields"][index]["source_logical_bytes"]+3*json.loads((ROOT/"source_snapshot.json").read_text())["fields"][index]["tensor_bytes"]+1024**2
        if charged_io()+reserved>P["conservative_total_io_budget_bytes"]:raise RuntimeError("cumulative I/O budget exhausted including failed attempts")
        event(dict(kind="field_reserved",attempt=attempt,index=index,reserved_application_bytes=reserved))
        child=subprocess.run([sys.executable,__file__,"one",str(index),attempt,revision_sha],timeout=min(800,3300-(time.monotonic()-start)))
        if child.returncode:raise RuntimeError("field child failed: "+str(index))
        results.append(json.loads((ROOT/"attempts"/attempt/("field%d.json"%index)).read_text()))
    report=dict(stage=number,attempt=attempt,status="validated",elapsed_seconds=time.monotonic()-start,resources_before=before,resources_after=resources(),fields=results,new_blob_bytes=sum(e["new_blob_bytes_this_attempt"] for e in results),application_payload_read_bytes=sum(e["application_payload_read_bytes"] for e in results),application_payload_written_bytes=sum(e["application_payload_written_bytes"] for e in results),originals_preserved=True)
    dest=ROOT/("stage%d_summary.json"%number)
    if dest.exists():
        previous=json.loads(dest.read_text())
        assert [(x["index"],x["source_sha256"],x["state_identity"]) for x in previous["fields"]]==[(x["index"],x["source_sha256"],x["state_identity"]) for x in results],"existing stage summary scope differs"
    else:write(dest,report)
    write(ROOT/"attempts"/attempt/"stage_summary.json",report)
    event(dict(kind="stage_validated",attempt=attempt,stage=number,elapsed_seconds=report["elapsed_seconds"],current_attempt_summary=str(ROOT/"attempts"/attempt/"stage_summary.json"),cumulative_charged_application_bytes=charged_io()));print(json.dumps(report),flush=True)
if __name__=="__main__":
    journal_safe=False
    try:
        if sys.argv[1]=="one":
            validate_plan_paths();journal_safe=True
            one(int(sys.argv[2]),sys.argv[3],sys.argv[4] if len(sys.argv)>4 else "")
        else:
            validate_plan_paths()
            ROOT.mkdir(parents=True,exist_ok=True)
            with (ROOT/".execution.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                journal_safe=True
                stage(int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else "")
    except Exception as exc:
        if journal_safe:event(dict(kind="failed",args=sys.argv[1:],error=repr(exc)))
        raise
