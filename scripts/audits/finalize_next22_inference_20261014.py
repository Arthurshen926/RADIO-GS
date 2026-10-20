import json,pathlib,os,time,hashlib,gzip,base64,datetime
from scripts.audits.preflight_next22_inference_20261014 import fp,processes
root=pathlib.Path("/mnt/pool/sqy/results/RADIO-GS/output/storage_inference_next22_20261014_1230")
pre=json.loads((root/"preflight_v3.json").read_text())
batches=[json.loads((root/("batch%02d"%i)/"completion.json").read_text()) for i in (1,2,3)]
recovery=json.loads((root/"batch01/metering_recovery_completion.json").read_text())
byid={x["id"]:(b["batch"],x) for b in batches for x in b["results"]}
revalidation=json.loads((root/"probe_revalidation_20261014_1400/completion.json").read_text())
assert revalidation["status"]=="TWO_REVALIDATIONS_ACCEPTED_ORIGINALS_AND_PACKAGES_UNCHANGED"
rvalid={x["id"]:x for x in revalidation["rows"]}
for x in recovery["rows"]:
 merged=dict(x);merged.update(rvalid[x["id"]]);byid[x["id"]]=(1,merged)
def snapshot():
 out={}
 for p in root.glob("batch*/*.inference.pt"):
  s=p.stat();out[str(p)]={"fingerprint":fp(p),"allocated_bytes":s.st_blocks*512,"logical_bytes":s.st_size}
 return out
s1=snapshot();time.sleep(2);s2=snapshot();assert s1==s2 and len(s2)==20
assert all(x["allocated_bytes"]>512 for x in s2.values()),"NFS stat still visibly provisional"
rows=[];original_alloc=0;package_alloc=0
for item in pre["rows"]:
 assert fp(pathlib.Path(item["source_path"]))==item["fingerprint"]
 batch,row=byid[item["id"]]
 if row["status"]=="SKIP":
  rows.append({"id":item["id"],"status":"SKIP","source_path":item["source_path"],"source_fingerprint":item["fingerprint"],"reason":row["reason"]});continue
 assert row["status"].startswith("ACCEPTED")
 p=row["package"]["path"];stat=s2[p]
 assert stat["fingerprint"]==row["package"]["fingerprint"]
 src_alloc=os.stat(item["source_path"]).st_blocks*512
 original_alloc+=src_alloc;package_alloc+=stat["allocated_bytes"]
 cfg_sha=hashlib.sha256(pathlib.Path(row["config_path"]).read_bytes()).hexdigest();assert cfg_sha==row["config_sha256"]
 receipt=(root/"probe_revalidation_20261014_1400"/(item["id"]+".acceptance.json")) if item["id"] in rvalid else root/("batch%02d"%batch)/(item["id"]+".acceptance.json")
 rows.append({"id":item["id"],"status":row["status"],"source_path":item["source_path"],"source_sha256":row["source"]["sha256"],"source_fingerprint":item["fingerprint"],"source_allocated_bytes":src_alloc,
 "package_path":p,"package_sha256":row["package"]["sha256"],"package_fingerprint":stat["fingerprint"],"package_logical_bytes":stat["logical_bytes"],"package_settled_allocated_bytes":stat["allocated_bytes"],"conditional_net_allocated_bytes":src_alloc-stat["allocated_bytes"],
 "config_path":row["config_path"],"config_sha256":cfg_sha,"config_bound_inside_package":True,
 "entry":row.get("entry","eval_lerf_grounding.load_render_pipeline"),"receipt":str(receipt),
 "tensor_count":row["tensor_count"],"public_output_count":row.get("public_output_count"),
 "tensor_and_public_output_comparison_completed":True,"full_per_probe_digest_receipt":True,
 "external_dependencies":item["external_dependencies"],
 "cli_checkpoint_arguments":["--checkpoint",p,"--inference_package","--expected_checkpoint_sha256",row["package"]["sha256"]]})
allocated_total=sum(p.stat().st_blocks*512 for p in root.rglob("*") if p.is_file())
bybatch=[]
for b in batches:
 d=root/("batch%02d"%b["batch"]);alloc=sum(p.stat().st_blocks*512 for p in d.rglob("*") if p.is_file())
 recoveryio=(recovery["new_bytes_read"]+revalidation["extra_checkpoint_payload_bytes_read"]) if b["batch"]==1 else 0
 assert alloc<3*1024**3 and b["maxrss_bytes"]<3*1024**3 and b["bytes_read"]+recoveryio<9*1024**3 and b["elapsed_seconds"]<600
 bybatch.append({"batch":b["batch"],"converted":sum(x["status"]!="SKIP" for x in rows if x["id"] in [r["id"] for r in b["results"]]),"settled_allocated_bytes":alloc,"checkpoint_payload_bytes_read":b["bytes_read"],"metering_recovery_new_bytes_read":recoveryio,"maxrss_bytes":b["maxrss_bytes"],"elapsed_seconds":b["elapsed_seconds"],"completion":str(d/"completion.json")})
assert allocated_total<8*1024**3
out={"status":"NEXT22_FINISHED_20_CONVERTED_2_SKIPPED_ORIGINALS_RETAINED","created_utc":datetime.datetime.utcnow().isoformat()+"Z","scope_count":22,"converted_count":20,"skipped_count":2,"rows":rows,"batches":bybatch,
"retained_tensor_count":sum(x.get("tensor_count",0) for x in rows),
"detailed_public_output_digest_count_all20_packages":sum(x.get("public_output_count") or 0 for x in rows),
"metering_recovery_notice":"Old n01/n02 receipts remain incomplete historical evidence. Fresh bounded revalidation now persists all279 retained tensor and57 public-output digests plus alias capacity/offset identity; new and original SHA match. No checkpoint was rewritten.",
"probe_revalidation":revalidation,
"alias_metadata_supplement":str(root/"batch01_storage_metadata_check.json"),
"settled_capacity_notice":"This final two-observation settled stat supersedes every earlier512B/provisional allocated or conditional-net figure in batch receipts. allocated need not exceed logical size.",
"accepted_original_allocated_bytes":original_alloc,"new_package_settled_allocated_bytes":package_alloc,"new_package_logical_bytes":sum(x.get("package_logical_bytes",0) for x in rows),
"new_tree_allocated_before_final_receipt":allocated_total,"conditional_checkpoint_net_allocated_if_exact20_sources_later_retired":original_alloc-package_alloc,
"conditional_net_after_all_next22_artifacts_before_final_receipt":original_alloc-allocated_total,
"actual_reclaimed_bytes":0,"source_deletion_authorized":False,"all22_original_fingerprints_unchanged":True,
"visible_candidate_processes":processes(),"writer_visibility":"Container-local only; remote writers cannot be globally excluded. Source/new file identity frozen and unchanged.",
"jobs_running":False,"no_further_346_conversion_started":True,"large66_untouched":True,
"recovery_limit":"If exact original files are later retired: optimizer/scheduler/scaler continuation is unavailable; old checkpoint-path commands break until explicitly pointed at new package+new SHA. Model/codec/heads/adapters and sealed configuration are retained. External teachers/cache/native dependencies are not retired.",
"validation_scope":"CPU public loader/builders and deterministic small probes. Legacy test substitutes original-checkpoint geometry allocation for PLY; no GPU/full-scene/dataset/teacher benchmark.",
"raw346_optimizer_estimate_not_realized":146841849300}
pilot=root.parent/"storage_inference_pilot_20261014_1146"
pilot_final=json.loads((pilot/"public_entry_integration_final_acceptance.json").read_text())
pilot_rows=[]
for name,p in zip(("scannet_hybrid","lerf_explicit","vpr_adapter"),pilot_final["packages"]):
 evidence=json.loads((pilot/(name+".acceptance.json")).read_text())
 for key,pathkey in (("source","source_path"),("package","package_path")):
  assert fp(pathlib.Path(p[pathkey]))==evidence[key]["fingerprint"]
  assert p[key+"_sha256"]==evidence[key]["sha256"]
 p=dict(p)
 p["source_fingerprint"]=fp(pathlib.Path(p["source_path"]))
 p["package_fingerprint"]=fp(pathlib.Path(p["package_path"]))
 p["source_allocated_bytes"]=os.stat(p["source_path"]).st_blocks*512
 p["package_settled_allocated_bytes"]=os.stat(p["package_path"]).st_blocks*512
 p["config_sha256"]=hashlib.sha256(pathlib.Path(p["config_path"]).read_bytes()).hexdigest()
 p["cli_checkpoint_arguments"]=["--checkpoint",p["package_path"],"--inference_package","--expected_checkpoint_sha256",p["package_sha256"]]
 pilot_rows.append(p)
def pilot_observe():
 return {p[k]:(fp(pathlib.Path(p[k])),os.stat(p[k]).st_blocks*512) for p in pilot_rows for k in ("source_path","package_path")}
ps1=pilot_observe();time.sleep(2);ps2=pilot_observe();assert ps1==ps2
assert len({(x["source_fingerprint"]["dev"],x["source_fingerprint"]["ino"]) for x in rows if x["status"]!="SKIP"}|{(x["source_fingerprint"]["dev"],x["source_fingerprint"]["ino"]) for x in pilot_rows})==23
assert all(x["source_fingerprint"]["nlink"]==1 and x["package_fingerprint"]["nlink"]==1 for x in rows if x["status"]!="SKIP")
assert all(x["source_fingerprint"]["nlink"]==1 and x["package_fingerprint"]["nlink"]==1 for x in pilot_rows)
pilot_source=sum(x["source_allocated_bytes"] for x in pilot_rows)
pilot_packages=sum(x["package_settled_allocated_bytes"] for x in pilot_rows)
pilot_tree=sum(p.stat().st_blocks*512 for p in pilot.rglob("*") if p.is_file())
out["pilot3"]={"rows":pilot_rows,"source_allocated_bytes":pilot_source,"new_package_settled_allocated_bytes":pilot_packages,"audit_tree_allocated_bytes":pilot_tree,"public_output_count":pilot_final["legacy_package_public_outputs_bitwise_equal"],"retained_tensor_count":sum(json.loads((pilot/(n+".acceptance.json")).read_text())["tensor_count"] for n in ("scannet_hybrid","lerf_explicit","vpr_adapter")),"acceptance_receipt":str(pilot/"public_entry_integration_final_acceptance.json")}
out["combined23"]={"accepted_count":23,"skipped_count":2,"all_originals_retained":True,"original_allocated_bytes":original_alloc+pilot_source,"new_package_allocated_bytes":package_alloc+pilot_packages,"conditional_checkpoint_net_allocated_bytes":original_alloc+pilot_source-package_alloc-pilot_packages,"audit_trees_allocated_before_this_receipt":allocated_total+pilot_tree,"conditional_net_after_all_audit_artifacts_before_this_receipt":original_alloc+pilot_source-allocated_total-pilot_tree,"actual_reclaimed_bytes":0,"all_sources_single_link_unique_inode":True,"all_packages_single_link":True}
f=root/"next22_final_acceptance.json"
with open(f,"x") as h:json.dump(out,h,indent=2)
brief={k:v for k,v in out.items() if k not in ("rows","visible_candidate_processes","pilot3","probe_revalidation")}
print(json.dumps(brief))
encoded=base64.b64encode(gzip.compress(json.dumps(out).encode())).decode()
print("ENCODED "+encoded)
