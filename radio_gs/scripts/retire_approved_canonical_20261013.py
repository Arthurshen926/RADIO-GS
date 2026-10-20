"""Delete exactly the approved six canonical payloads after revalidation."""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
import torch
from radio_gs.scripts.migrate_capability_storage import fingerprint, file_sha256
from radio_gs.scripts.finalize_capability_pilot import metadata_only
from radio_gs.interfaces.capability_cache import load_canonical_capability_bank, load_canonical_support_graph

ROOT = Path("/mnt/pool/sqy/results/RADIO-GS/output")
MANIFEST = Path("/root/RADIO-GS-query-field/docs/experiments/2026-10-12-five-fern-retirement-candidates.json")
MANIFEST_SHA = "81480c27aec7103866d17d81637641301b582e69f4a3b6716f5bf7d4ffa1e95c"
NAMES = [
"nvos_fern_compact_d192_l96_caploss_official_dino_sam3_views.pt",
"nvos_fern_compact_d192_l96_official_dino_sam3_views.pt",
"nvos_fern_compact_d192_l96_sam1_caploss_official_dino_sam3_views.pt",
"nvos_fern_compact_d256_l128_caploss_official_dino_sam3_views.pt",
"nvos_fern_d384_verified_pose_official_dino_sam3_views.pt"]
ALLOWED = {str(ROOT/"canonical_query_features"/n) for n in NAMES}
ALLOWED.add(str(ROOT/"storage_migration_20261012/fern_caploss_v1"/NAMES[0]))

def info(path):
    p = Path(path)
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or p.resolve(strict=True) != p:
        raise RuntimeError("not exact regular physical path: " + str(p))
    return dict(path=str(p), fingerprint=fingerprint(p), logical=s.st_size,
                allocated=s.st_blocks*512, nlink=s.st_nlink)

def unused(paths):
    identities={(Path(p).stat().st_dev,Path(p).stat().st_ino) for p in paths}
    start=time.monotonic(); checked=0; read_bytes=0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():
            continue
        if time.monotonic()-start>15:
            raise RuntimeError("process usage inspection budget exhausted")
        try:
            for fd in (proc/"fd").iterdir():
                try:
                    s=fd.stat()
                except FileNotFoundError:
                    continue
                if (s.st_dev,s.st_ino) in identities:
                    raise RuntimeError("source open by local PID " + proc.name)
            with (proc/"maps").open() as handle:
                for line in handle:
                    read_bytes+=len(line)
                    if read_bytes>64*1024**2:
                        raise RuntimeError("process map inspection budget exhausted")
                    fields=line.split(None,5)
                    if len(fields)>=5 and int(fields[4]):
                        major,minor=(int(v,16) for v in fields[3].split(":"))
                        if (os.makedev(major,minor),int(fields[4])) in identities:
                            raise RuntimeError("source mapped by local PID " + proc.name)
            checked+=1
        except (FileNotFoundError,ProcessLookupError):
            continue
    return {"processes_checked":checked,"map_metadata_bytes":read_bytes,
            "scope":"eden local fd/maps only; distributed host users coordinated separately"}

def main():
    torch.set_num_threads(2)
    assert file_sha256(MANIFEST) == MANIFEST_SHA
    m = json.loads(MANIFEST.read_text())
    targets = [dict(r["source"], sha256=r["source_sha256"]) for r in m["records"]]
    targets.append(m["additional_redundant_pilot"])
    assert len(targets) == 6 and {r["path"] for r in targets} == ALLOWED
    journal_dir = ROOT/"storage_migration_20261012/retirement_20261013_canonical"
    journal_dir.mkdir(exist_ok=False)
    journal_path = journal_dir/"journal.json"
    j = dict(schema="radio_gs.approved_canonical_retirement.v1", status="validating",
             approval_message="Sentinel_b244900e1a78819194b7ab6816438b40",
             candidate_manifest=str(MANIFEST), candidate_manifest_sha256=MANIFEST_SHA,
             started_unix=time.time(), targets=targets, validation=[], deleted=[])
    def save():
        temp = journal_path.with_suffix(".tmp")
        with temp.open("w") as h:
            json.dump(j,h,indent=2);h.write("\n");h.flush();os.fsync(h.fileno())
        os.replace(temp,journal_path)
        fd=os.open(journal_dir,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    save()
    try:
        protected = {}
        for r in m["records"]:
            new = r["new"]; p=Path(new["path"])
            actual=info(p)
            assert actual["fingerprint"] == new["fingerprint"]
            assert file_sha256(p) == r["new_sha256"]
            assert file_sha256(r["receipt"]) == r["receipt_sha256"]
            receipt=json.loads(Path(r["receipt"]).read_text())
            assert receipt["status"] == "validated"
            side=p.with_suffix(".pt.json")
            side_metadata=json.loads(side.read_text())
            for key,value in metadata_only(p).items():
                assert side_metadata[key]==value
            original_receipt=receipt
            if "parent_receipt" in receipt:
                assert file_sha256(receipt["parent_receipt"])==receipt["parent_receipt_sha256"]
                original_receipt=json.loads(Path(receipt["parent_receipt"]).read_text())
                protected[receipt["parent_receipt"]]=info(receipt["parent_receipt"])
            assert original_receipt["source"]==r["source"]["path"]
            assert file_sha256(Path(r["source"]["path"]).with_suffix(".pt.json"))==original_receipt["source_sidecar_sha256"]
            protected[str(side)] = info(side)
            protected[r["receipt"]] = info(r["receipt"])
            protected[str(p)] = actual
            bank=load_canonical_capability_bank(p)
            assert bank.features_are_compact and len(bank.global_rows)==297200
            graph_paths = [receipt["support_graph"]] if "support_graph" in receipt else list(receipt.get("existing_support_graphs",{}))
            for graph_path in graph_paths:
                graph=load_canonical_support_graph(graph_path,bank)
                protected[graph_path]=info(graph_path)
                del graph
            del bank;gc.collect()
            j["validation"].append(dict(new=str(p),fresh_sha256=r["new_sha256"],
                compact_reader_pass=True,existing_graphs_composed=len(graph_paths)))
            save()
        for r in targets:
            actual=info(r["path"])
            assert actual["fingerprint"]==r["fingerprint"] and actual["nlink"]==1
            assert file_sha256(r["path"])==r["sha256"]
            side=Path(r["path"]).with_suffix(".pt.json")
            protected[str(side)]=info(side)
        for r in targets:
            assert info(r["path"])["fingerprint"]==r["fingerprint"]
        j["local_usage_check"]=unused(sorted(ALLOWED))
        j["before"]=[info(r["path"]) for r in targets]
        j["df_before"]=shutil.disk_usage(ROOT)._asdict()
        j["status"]="validated_before_unlink";save()
        for r in targets:
            # All six are pre-approved; this loop cannot accept arbitrary paths.
            assert r["path"] in ALLOWED
            a=info(r["path"])
            assert a["fingerprint"]==r["fingerprint"] and a["nlink"]==1
            unused([r["path"]])
            for p, prior in protected.items():
                assert info(p)["fingerprint"]==prior["fingerprint"]
            os.unlink(r["path"])
            assert not Path(r["path"]).exists()
            j["deleted"].append(a);save()
        j["df_after"]=shutil.disk_usage(ROOT)._asdict()
        j["protected_retained"]=[info(p) for p in protected]
        j["deleted_logical_bytes"]=sum(r["logical"] for r in j["deleted"])
        j["deleted_allocated_bytes_before_unlink"]=sum(r["allocated"] for r in j["deleted"])
        j["observed_mount_free_delta"]=j["df_after"]["free"]-j["df_before"]["free"]
        j["allocation_note"]="file allocated accounting and mount df delta are distinct; NFS df may lag or include unrelated activity"
        j.update(status="completed",ended_unix=time.time())
        save();print(json.dumps(j,indent=2),flush=True)
    except Exception as e:
        j.update(status="stopped",error=repr(e),ended_unix=time.time());save();raise

if __name__=="__main__":
    main()

