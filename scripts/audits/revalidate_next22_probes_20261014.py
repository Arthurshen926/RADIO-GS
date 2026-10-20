"""Revalidate exactly n01/n02 existing packages; no checkpoint writes or deletion."""
import gc,hashlib,json,os,resource,shutil,threading,time,datetime
from pathlib import Path
import torch
from scripts.audits import convert_next22_inference_20261014 as c
from radio_gs.utils.inference_checkpoint import DROP_KEYS,read_project_checkpoint_once,fingerprint
from radio_gs.utils.evaluation_checkpoint import InferencePackageReader,evaluation_package_identity

ROOT=c.ROOT
DEST=ROOT/"probe_revalidation_20261014_1400"
START=time.monotonic();READ=0;STOP=threading.Event()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def check():
 assert time.monotonic()-START<600
 assert READ<2*1024**3
 assert c.allocated(DEST)<16*1024**2
 assert c.allocated(ROOT)<8*1024**3
def guard():
 while not STOP.wait(.1):
  rss=int(next(s.split()[1] for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmRSS:")))*1024
  try:
   assert rss<3*1024**3
   check()
  except Exception:os._exit(90)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not torch.cuda.is_available()
assert shutil.disk_usage(ROOT).free>20*1024**3
pre=json.loads((ROOT/"preflight_v3.json").read_text())
recovery=json.loads((ROOT/"batch01/metering_recovery_completion.json").read_text())
recovered={r["id"]:r for r in recovery["rows"]}
selected=[x for x in pre["rows"] if x["id"] in ("n01","n02")]
assert len(selected)==2
assert sum(x["archive_bytes"]+recovered[x["id"]]["package"]["bytes_read"] for x in selected)<2*1024**3
DEST.mkdir(exist_ok=False)
sources=c.runtime_sources(Path.cwd());c.dump(DEST/"runtime_source_snapshot.json",sources)
threading.Thread(target=guard,daemon=True).start()
results=[]
try:
 for item in selected:
  r=recovered[item["id"]]
  c.source_stable(item)
  cfgpath=Path(r["config_path"]);raw=cfgpath.read_bytes()
  assert hashlib.sha256(raw).hexdigest()==r["config_sha256"]
  cfg=json.loads(raw);assert cfg==item["config"]
  old,record=read_project_checkpoint_once(item["source_path"],expected_sha256=r["source"]["sha256"],expected_fingerprint=item["fingerprint"])
  READ+=record["bytes_read"]
  expected={k:v for k,v in old.items() if k not in DROP_KEYS}
  del old;gc.collect()
  torch.manual_seed(0)
  modules,entry=c.build_modules(expected,cfg,cfgpath,item["source_path"])
  old_outputs=c.probes(expected,cfg,loaded_modules=modules)
  del modules;gc.collect()
  path=Path(r["package"]["path"])
  assert fingerprint(os.stat(path,follow_symlinks=False))==r["package"]["fingerprint"]
  reader=InferencePackageReader(path,expected_sha256=r["package"]["sha256"])
  loaded=reader.load(path);READ+=reader.record["bytes_read"]
  assert reader.provenance["source_checkpoint"]["sha256"]==record["sha256"]
  assert reader.provenance["config_record"]["config"]==cfg
  for p,sha in reader.provenance["runtime_source_sha256"].items():
   if p.startswith("radio_gs/"):assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,("runtime changed",p)
  tensors=c.compare(expected,loaded)
  aliases=c.alias_signature(expected);assert aliases==c.alias_signature(loaded)
  assert c.alias_groups(expected)==c.alias_groups(loaded)
  torch.manual_seed(0)
  modules,newentry=c.build_modules(loaded,cfg,cfgpath,path,reader)
  new_outputs=c.probes(loaded,cfg,loaded_modules=modules)
  checks=c.compare(old_outputs,new_outputs);assert entry==newentry
  c.source_stable(item)
  assert fingerprint(os.stat(path,follow_symlinks=False))==r["package"]["fingerprint"]
  assert hashlib.sha256(cfgpath.read_bytes()).hexdigest()==r["config_sha256"]
  check()
  result={"id":item["id"],"status":"ACCEPTED_REVALIDATED_ORIGINAL_RETAINED","created_utc":datetime.datetime.utcnow().isoformat()+"Z",
   "source":record,"package":reader.record,"config_path":str(cfgpath),"config_sha256":r["config_sha256"],"entry":entry,
   "identity":evaluation_package_identity(reader),"tensor_count":len(tensors),"tensor_manifest":tensors,"tensor_manifest_sha256":digest(tensors),
   "storage_capacity_offset_signature":aliases,"alias_signature_sha256":digest(aliases),"alias_topology_equal":True,
   "public_output_count":len(checks),"public_output_manifest":checks,"public_output_manifest_sha256":digest(checks),
   "outputs_bitwise_equal":True,"max_abs_error":0,"relative_linf_error":0,
   "same_config_seed_threads_dtype_eval":True,"seed":0,"torch_num_threads":2,"interop_threads":1,
   "source_bytes_read":record["bytes_read"],"package_bytes_read":reader.record["bytes_read"],
   "original_and_package_fingerprints_unchanged":True,"weights_rewritten":False,
   "old_incomplete_receipt_retained":str(ROOT/"batch01"/(item["id"]+".metering_recovery.json")),
   "scope":"Fresh CPU real public builder/probe comparison; original-checkpoint geometry allocation substitutes external PLY. No renderer/dataset/teacher/GPU benchmark."}
  c.dump(DEST/(item["id"]+".acceptance.json"),result)
  results.append({k:v for k,v in result.items() if k not in ("tensor_manifest","storage_capacity_offset_signature","public_output_manifest")})
  print(json.dumps(results[-1]),flush=True)
  del expected,loaded,reader,modules,old_outputs,new_outputs,tensors,checks,aliases,result;gc.collect()
 assert c.runtime_sources(Path.cwd())==sources
 check()
 out={"status":"TWO_REVALIDATIONS_ACCEPTED_ORIGINALS_AND_PACKAGES_UNCHANGED","rows":results,
 "extra_checkpoint_payload_bytes_read":READ,"maxrss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 "elapsed_seconds":time.monotonic()-START,"receipt_tree_allocated_bytes":c.allocated(DEST),
 "runtime_sources_unchanged":True,"actual_reclaimed_bytes":0,"limits":{"read":2*1024**3,"rss":3*1024**3,"seconds":600,"receipt_tree":16*1024**2}}
 c.dump(DEST/"completion.json",out)
 print(json.dumps({k:v for k,v in out.items() if k!="rows"}),flush=True)
finally:STOP.set()
