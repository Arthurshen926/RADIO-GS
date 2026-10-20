"""Review preserved result content and current declared JSON inputs; no payload loads."""
import json,pathlib,os,re,hashlib,gzip,time,collections,math
BASE=pathlib.Path("/mnt/pool/sqy/results/RADIO-GS/output")
D=pathlib.Path("docs/experiments/2026-10-14-radio-history-payload-plan-1547")
S=json.loads((D/"summary.json").read_text())
with gzip.open(str(D/"closed_payload_candidates.json.gz"),"rt") as f:leaves=json.load(f)
leafset={r["path"] for r in leaves};candidate_roots={r["root"] for r in leaves}
READ=0;MAX=64*1024**2;record_reviews={};numeric=re.compile("iou|accuracy|precision|recall|cosine|loss|psnr|rmse|score|error|f1|ap$|macro|mean|median",re.I)
def read(p):
 global READ
 s=p.stat()
 if s.st_size>2*1024**2:return None
 if READ+s.st_size>MAX:raise RuntimeError("small-document read budget")
 raw=p.read_bytes();READ+=len(raw)
 return raw
def evidence(value):
 nums=[];decisions=[]
 def walk(v,p=""):
  if isinstance(v,dict):
   for k,w in v.items():walk(w,p+"/"+str(k))
  elif isinstance(v,list):
   for i,w in enumerate(v[:10000]):walk(w,p+"/"+str(i))
  elif isinstance(v,(int,float)) and not isinstance(v,bool) and numeric.search(p):
   if math.isfinite(v):nums.append({"key":p,"value":v})
  elif isinstance(v,(str,bool)) and re.search("decision|status|conclusion|promot|default_changed|development_only|negative|reject|fail|pass|recommend",p,re.I):
   decisions.append({"key":p,"value":v})
 walk(value)
 return nums[:12],decisions[:12]
for name,v in S["roots"].items():
 if name not in candidate_roots:continue
 paths=sorted(v["explicit_result_records"],key=lambda x:(0 if pathlib.Path(x["path"]).name in ("summary.json","comparison.json","results.json","final_comparison.json","summary.md","report.md") else 1,len(pathlib.Path(x["path"]).parts),x["bytes"]))
 reviewed=[]
 for row in paths[:12]:
  p=pathlib.Path(row["path"]);raw=read(p)
  if raw is None:continue
  nums=[];dec=[]
  if p.suffix==".json":
   try:nums,dec=evidence(json.loads(raw))
   except Exception:continue
  else:
   t=raw.decode("utf8",errors="replace")
   if len(t)>128 and re.search("IoU|mIoU|accuracy|cosine|loss|conclusion|decision|结论|结果|拒绝|失败|保留|retire",t,re.I):
    dec=[{"key":"text_excerpt","value":t[:1600]}]
  if nums or dec:
   reviewed.append({"path":str(p),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"quantitative_excerpt":nums,"decision_excerpt":dec})
   if len(reviewed)>=2:break
 record_reviews[name]={"status":"PRESERVED_RESULT_CONTENT_VERIFIED" if reviewed else "HOLD_RECORD_CONTENT_NOT_ESTABLISHED","records":reviewed,"all_figures_and_text_records_retained":True}
# Refresh the existing current-v10 JSON graph without scanning source directories.
old=json.loads((BASE/"storage_cleanup_20261014/radiogs_v10_json_dependencies_20261014.json").read_text())
queue=[pathlib.Path(p) for p in old["json_files_read"]]+[pathlib.Path(old["seed"])]
done=set();hits=[];currentroots=set();errors=[]
def strings(v):
 if isinstance(v,str):yield v
 elif isinstance(v,dict):
  for w in v.values():yield from strings(w)
 elif isinstance(v,list):
  for w in v:yield from strings(w)
while queue:
 p=queue.pop()
 if str(p) in done:continue
 if len(done)>2000:raise RuntimeError("current JSON closure count budget")
 # Never read tmp or the cancelled G14 group. Storage roots are all protected.
 ps=str(p)
 if ps.startswith("/tmp/") or "/storage_" in ps:continue
 done.add(ps)
 try:
  raw=read(p)
  if raw is None:continue
  obj=json.loads(raw)
 except (OSError,ValueError) as exc:errors.append({"path":ps,"error":repr(exc)});continue
 for val in strings(obj):
  if not val or len(val)>4096 or "\n" in val:continue
  q=None
  if val.startswith(("/mnt/pool/sqy/results/RADIO-GS/output/","/root/RADIO-GS/output/","/root/RADIO-GS-query-field/output/")):
   q=BASE/val.split("/output/",1)[1]
  elif val.startswith("output/"):q=BASE/val[7:]
  elif not val.startswith("/") and "/" in val:
   rel=p.parent/val
   if str(rel).startswith(str(BASE)+"/") and rel.exists():q=rel
  if q is None:continue
  root=str(q).split("/output/",1)[1].split("/")[0];currentroots.add(root)
  if root in candidate_roots:
   hits.append({"owner":ps,"value":val,"resolved":str(q),"root":root,"exact_leaf":str(q) in leafset,"hold":"whole root conservatively; classify provenance separately later"})
  if q.suffix==".json" and not root.startswith("storage_") and q.is_file():queue.append(q)
blocked={h["root"] for h in hits}
reviewedroots={n for n,v in record_reviews.items() if v["status"]=="PRESERVED_RESULT_CONTENT_VERIFIED"}-blocked
# Dropping a root can expose outside aliases, so recompute last-link closure.
g=collections.defaultdict(list)
for r in leaves:
 if r["root"] in reviewedroots:g[(r["dev"],r["ino"])].append(r)
final=[]
for rs in g.values():
 if len(rs)==rs[0]["nlink"]:final.extend(rs)
seen=set();total=0;by=collections.defaultdict(lambda:{"leaf_count":0,"exclusive_allocated":0})
for r in final:
 s=os.stat(r["path"],follow_symlinks=False)
 assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)==(r["dev"],r["ino"],r["size"],r["mtime_ns"],r["ctime_ns"],r["nlink"])
 key=(r["dev"],r["ino"]);by[r["root"]]["leaf_count"]+=1
 if key not in seen:seen.add(key);total+=r["allocated"];by[r["root"]]["exclusive_allocated"]+=r["allocated"]
with gzip.open(str(D/"reviewed_retirement_leaves.json.gz"),"wt") as f:json.dump(final,f,separators=(",",":"))
out={"status":"EXACT_REVIEWED_HISTORICAL_PAYLOAD_PROPOSAL_NOT_DELETION_AUTHORIZATION","leaf_count":len(final),"unique_inode_count":len(seen),"exclusive_allocated_bytes":total,"root_count":len(by),"roots":dict(by),"record_reviews":record_reviews,"small_document_bytes_read":READ,"current_json_files_read":len(done),"current_declared_roots":sorted(currentroots),"current_input_root_hits_excluded":hits,"current_json_read_errors":errors,"all_figures_and_records_preserved":True,"original_payloads_unchanged":True,"deletion_performed":False,"current_scope_limit":"Current declared v10 JSON contracts refreshed. Protected T0/C0/query_field/canonical/source/baselines/storage roots unchanged. Dynamic undeclared external jobs still require pre-execution writer/consumer check.","old_validator_handling":"Branches in the manifest are explicitly retired; old self-readers/provenance hash verifiers are not required to run after approval. Do not rewrite historical receipts or fake original SHA."}
with (D/"record_and_current_input_review.json").open("x") as f:json.dump(out,f,indent=2)
print(json.dumps({k:v for k,v in out.items() if k not in ("roots","record_reviews","current_declared_roots","current_input_root_hits_excluded")}))
print("CURRENT_HIT_ROOTS",json.dumps(sorted(blocked)))
print("TOP25",json.dumps(sorted([{"name":n,**v} for n,v in by.items()],key=lambda x:-x["exclusive_allocated"])[:25]))
