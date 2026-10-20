"""Read-only payload retirement inventory under explicit historical-records contract."""
import os,re,json,pathlib,stat,time,collections,gzip,hashlib,datetime
BASE=pathlib.Path("/mnt/pool/sqy/results/RADIO-GS/output")
DEST=pathlib.Path("docs/experiments/2026-10-14-radio-h449-payload-plan-1635")
assert not DEST.exists()
inv=json.loads(pathlib.Path("docs/experiments/2026-10-14-radio-h449-input-roots-1635.json").read_text())
old=json.loads(pathlib.Path("docs/experiments/2026-10-14-radio-history-payload-plan-1547/final_retirement_summary.json").read_text())
protected={n:["current_v10_fresh_JSON_declared_root_hold"] for n in old["current_declared_roots"]}
roots=inv["roots"]
assert len(roots)==449 and len({r["path"] for r in roots})==449
assert all(pathlib.Path(r["path"]).parent==BASE for r in roots)
selected=[r for r in roots if pathlib.Path(r["path"]).name not in protected]
assert len(selected)<500
DEST.mkdir()
start=time.time();errors=[];raw=[];records=[];per={};record_rg=re.compile(r"(compar|summary|result|metric|report|readout|decision|validation|conclusion)",re.I)
payload={".pt",".pth",".ckpt",".safetensors",".npy",".npz",".f16",".bin",".ply"}
texts={".json",".jsonl",".md",".txt",".csv",".tsv",".log",".yaml",".yml",".py",".sh",".toml",".html",".pdf"}
device=BASE.stat().st_dev
for entry in selected:
 name=entry["path"].split("/")[-1];root=pathlib.Path(entry["path"])
 info={"path":str(root),"root_attributed_bytes_old_snapshot":entry["allocated_bytes"],"regular_files":0,"candidate_files":0,"record_files":0,"explicit_result_records":[],"figures_retained":0,"symlinks_retained":0,"unclassified_retained":0,"errors":[]}
 if not stat.S_ISDIR(os.lstat(root).st_mode):
  info["errors"].append({"path":str(root),"error":"root is not a direct directory; do not follow root symlink"});per[name]=info;continue
 stack=[root]
 while stack:
  if time.time()-start>600:raise RuntimeError("inventory time budget600s")
  d=stack.pop()
  try:children=list(os.scandir(str(d)))
  except OSError as e:info["errors"].append({"path":str(d),"error":repr(e)});continue
  for e in children:
   try:s=e.stat(follow_symlinks=False)
   except OSError as exc:info["errors"].append({"path":e.path,"error":repr(exc)});continue
   if s.st_dev!=device:info["unclassified_retained"]+=1;continue
   if stat.S_ISLNK(s.st_mode):info["symlinks_retained"]+=1;continue
   if stat.S_ISDIR(s.st_mode):stack.append(pathlib.Path(e.path));continue
   if not stat.S_ISREG(s.st_mode):continue
   info["regular_files"]+=1;p=pathlib.Path(e.path);suffix=p.suffix.lower()
   if suffix in texts:
    info["record_files"]+=1
    if s.st_size>=64 and suffix in {".json",".md",".csv",".tsv",".txt",".html"} and record_rg.search(p.name):
     info["explicit_result_records"].append({"path":str(p),"bytes":s.st_size})
    continue
   if suffix in {".png",".jpg",".jpeg",".svg",".gif",".webp",".mp4",".mov"}:
    info["figures_retained"]+=1;continue
   if suffix not in payload or s.st_size<1024**2:
    info["unclassified_retained"]+=1;continue
   # Keep small arrays likely containing quantitative results; big discarded
   # predictions lose old rescoring capability, covered explicitly by retirement.
   if suffix in {".npy",".npz"} and s.st_size<64*1024**2 and record_rg.search(str(p.relative_to(root))):
    info["unclassified_retained"]+=1;continue
   raw.append({"path":str(p),"root":name,"size":s.st_size,"allocated":s.st_blocks*512,"dev":s.st_dev,"ino":s.st_ino,"nlink":s.st_nlink,"mode":s.st_mode,"mtime_ns":s.st_mtime_ns,"ctime_ns":s.st_ctime_ns})
   info["candidate_files"]+=1
 per[name]=info
known_external={}
eligible_roots={n for n,v in per.items() if not v["errors"] and (v["explicit_result_records"] or n in known_external)}
for n,v in per.items():
 v["records_status"]="HAS_RESULT_RECORD_CANDIDATES_REQUIRES_CONTENT_REVIEW" if v["explicit_result_records"] else ("EXTERNAL_CONCLUSION_ALREADY_AUDITED" if n in known_external else "HOLD_NO_RESULT_RECORD_IDENTIFIED")
 v["external_evidence"]=known_external.get(n,[])
groups=collections.defaultdict(list)
for r in raw:
 if r["root"] in eligible_roots:groups[(r["dev"],r["ino"])].append(r)
closed=[];cross=[]
for key,rs in groups.items():
 ns={r["nlink"] for r in rs};bs={r["allocated"] for r in rs}
 if len(ns)!=1 or len(bs)!=1 or len(rs)!=next(iter(ns)):cross.extend(rs)
 else:closed.extend(rs)
for r in closed:
 s=os.stat(r["path"],follow_symlinks=False)
 assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)==(r["dev"],r["ino"],r["size"],r["mtime_ns"],r["ctime_ns"],r["nlink"]),"candidate changed"
by=collections.defaultdict(lambda:{"leaf_count":0,"exclusive_allocated_bytes":0,"shared_closed_inode_count":0})
seen=set()
for r in closed:
 v=by[r["root"]];v["leaf_count"]+=1;key=(r["dev"],r["ino"])
 if key not in seen:v["exclusive_allocated_bytes"]+=r["allocated"];seen.add(key);v["shared_closed_inode_count"]+=int(r["nlink"]>1)
for n,v in per.items():v["closed_candidate"]=by[n]
with gzip.open(str(DEST/"closed_payload_candidates.json.gz"),"wt") as f:json.dump(closed,f,separators=(",",":"))
with gzip.open(str(DEST/"all_payload_inventory.json.gz"),"wt") as f:json.dump(raw,f,separators=(",",":"))
with gzip.open(str(DEST/"external_or_retained_hardlinks_hold.json.gz"),"wt") as f:json.dump(cross,f,separators=(",",":"))
summary={"status":"READ_ONLY_H449_PROPOSAL_REQUIRES_RESULT_CONTENT_AND_CURRENT_T0_C0_DEPENDENCY_REVIEW_AND_NEW_APPROVAL","created_utc":datetime.datetime.utcnow().isoformat()+"Z","elapsed_seconds":time.time()-start,"scanned_roots":len(selected),"protected_roots":[{"root":r["path"],"allocated_attribution":r["allocated_bytes"],"reasons":sorted(set(protected[r["path"].split("/")[-1]]))} for r in roots if r["path"].split("/")[-1] in protected],"roots":per,"total_payload_files":len(raw),"closed_candidate_leaves":len(closed),"closed_unique_inodes":len(seen),"exclusive_allocated_bytes":sum(v["exclusive_allocated_bytes"] for v in by.values()),"held_cross_or_retained_link_leaves":len(cross),"all_candidate_fingerprints_stable_on_second_pass":True,"payload_bytes_read":0,"historical_records_and_all_figures_retained":True,"no_deletions":True,"no_runtime_changes":True,"retired_capability_if_approved":"Exact old branch weights,teacher arrays,old predictions and training resume may cease to exist. Old branch entrypoints/hash validators must report retired-by-policy rather than silently substitute.","cancellation_scope":"G14 manifest not read; allstorage roots untouched","accounting":"Exclusive within candidate list: all nlink names found inside candidate; outside/retained aliases held. Live external writers not globally excluded; fresh execution precheck still required."}
with (DEST/"summary.json").open("x") as f:json.dump(summary,f,indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ("roots","protected_roots")}))
print("PROTECTED",json.dumps(summary["protected_roots"]))
print("TOP25",json.dumps(sorted([{"name":n,**v["closed_candidate"],"explicit_results":len(v["explicit_result_records"]),"records_status":v["records_status"]} for n,v in per.items()],key=lambda x:-x["exclusive_allocated_bytes"])[:25]))
print("ARTIFACT",str(DEST))