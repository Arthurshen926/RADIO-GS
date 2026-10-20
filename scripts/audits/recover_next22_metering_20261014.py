"""Recover two metering-only receipts without rewriting any checkpoint."""
import json,pathlib,os,io,zipfile,hashlib,time
from scripts.audits.preflight_next22_inference_20261014 import Metadata,fp
root=pathlib.Path("/mnt/pool/sqy/results/RADIO-GS/output/storage_inference_next22_20261014_1230")
metadata=json.loads(pathlib.Path(__file__).with_name("next22_batch01_metadata_20261014.json").read_text())
with open(root/"batch01_storage_metadata_check.json","x") as f:json.dump(metadata,f,indent=2)
pre=json.loads((root/"preflight_v3.json").read_text());items={x["id"]:x for x in pre["rows"]}
batch=json.loads((root/"batch01/completion.json").read_text())
snapshot=json.loads((root/"batch01/runtime_source_snapshot.json").read_text())
source=next(x for x in snapshot if x["path"]=="scripts/audits/convert_next22_inference_20261014.py")
assert hashlib.sha256(source["text"].encode()).hexdigest()==source["sha256"]
for gate in ["tensors=compare(expected,loaded)","output_checks=compare(old_outputs,new_outputs)","assert alias_groups(expected)==alias_groups(loaded)"]:
 assert source["text"].index(gate)<source["text"].index('raise RuntimeError("NFS allocation did not settle")')
rows=[]
for ident in ("n01","n02"):
 skip=next(x for x in batch["results"] if x["id"]==ident)
 assert skip["reason"]=="RuntimeError('NFS allocation did not settle')"
 item=items[ident];path=root/"batch01"/(ident+".inference.pt")
 assert fp(pathlib.Path(item["source_path"]))==item["fingerprint"]
 before=fp(path);h=hashlib.sha256();read=0
 with open(path,"rb") as f:
  while True:
   data=f.read(8*1024**2)
   if not data:break
   h.update(data);read+=len(data)
 assert fp(path)==before
 with zipfile.ZipFile(path) as z:
  name=next(x for x in z.namelist() if x.endswith("/data.pkl"))
  value=Metadata(io.BytesIO(z.read(name))).load()
 assert value["schema"]=="radiogs.inference_checkpoint.v1" and value["resume_capable"] is False
 record=value["provenance"]["source_checkpoint"]
 assert record["fingerprint"]==item["fingerprint"] and record["path"]==item["source_path"]
 meta=next(x for x in metadata["rows"] if x["id"]==ident);assert meta["status"]=="PASS"
 a1=path.stat().st_blocks*512;time.sleep(1);a2=path.stat().st_blocks*512;assert a1==a2 and a2>0
 row={"status":"ACCEPTED_METERING_RECOVERY_ORIGINAL_RETAINED","id":ident,"source":record,
      "source_allocated_bytes":os.stat(item["source_path"]).st_blocks*512,
      "package":{"path":str(path),"sha256":h.hexdigest(),"fingerprint":before,"bytes_read":read},
      "package_allocated_bytes":a2,"package_logical_bytes":before["size"],
      "config_path":str(root/"batch01"/(ident+".config.json")),
      "config_sha256":hashlib.sha256((root/"batch01"/(ident+".config.json")).read_bytes()).hexdigest(),
      "tensor_count":meta["tensor_count"],"tensor_values_shape_dtype_stride_verified_during_conversion":True,
      "storage_capacity_offset_alias_crc_metadata_equal":True,"public_outputs_bitwise_equal_during_conversion":True,
      "numeric_equality_evidence":{"type":"completed conversion gates before metering-only exception",
        "converter_source_sha256":source["sha256"],"batch_completion":str(root/"batch01/completion.json"),
        "old_failure_preserved":str(root/"batch01"/(ident+".skip.json"))},
      "per_tensor_and_per_probe_digest_manifest_saved_before_failure":False,
      "metadata_check":str(root/"batch01_storage_metadata_check.json"),"recovery_new_bytes_read":read,
      "recovery_original_payload_bytes_read":0,"source_stat_unchanged":True,"new_file_stat_unchanged":True,
      "actual_reclaimed_bytes":0,
      "note":"Only NFS metering recovered; no checkpoint rewritten, no numeric threshold changed. CRC metadata supplements the original completed tensor/public-output comparisons; per-probe digests were not saved before the metering exception."}
 with open(root/"batch01"/(ident+".metering_recovery.json"),"x") as f:json.dump(row,f,indent=2)
 rows.append(row)
out={"status":"METERING_RECOVERED","rows":rows,"new_bytes_read":sum(x["recovery_new_bytes_read"] for x in rows),"original_payload_bytes_read":0}
with open(root/"batch01/metering_recovery_completion.json","x") as f:json.dump(out,f,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!="rows"}))
