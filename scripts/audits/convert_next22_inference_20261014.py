"""Exact-scope, non-destructive next22 inference conversion. Originals retained."""
from __future__ import annotations
import argparse,gc,hashlib,json,os,resource,shutil,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from radio_gs.utils.inference_checkpoint import DROP_KEYS,fingerprint,read_project_checkpoint_once,make_inference_package
from radio_gs.utils.evaluation_checkpoint import InferencePackageReader,initialize_package_geometry,package_use_2dgs,evaluation_package_identity
from radio_gs.scripts import eval_scannet_pointcloud_radio_gs as scan
from radio_gs.scripts import eval_lerf_grounding as ground
from radio_gs.scripts import eval_lerf_direct_3d_selection as direct
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from scripts.audits.inference_pilot_probes import probes
from scripts.audits.pilot_inference_checkpoint_20261014 import compare,alias_groups,tensor_bytes,runtime_sources
from scripts.audits.preflight_next22_inference_20261014 import processes

ROOT=Path("/mnt/pool/sqy/results/RADIO-GS/output/storage_inference_next22_20261014_1230")
PLAN=Path(__file__).with_name("next22_inference_plan_20261014.json")
GIB=1024**3;RAM_LIMIT=3*GIB;BATCH_DISK=3*GIB;TOTAL_DISK=8*GIB;READ_LIMIT=9*GIB
START=0;BYTES_READ=0;WRITTEN=0;DEST=None;STOP=threading.Event()

def allocated(root):
    return sum(p.stat().st_blocks*512 for p in root.rglob("*") if p.is_file())

def dump(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    assert len(raw)<32*1024**2
    with open(path,"xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())

def limits():
    assert time.monotonic()-START<600,"batch time budget exceeded"
    assert BYTES_READ<=READ_LIMIT,"read IO budget exceeded"
    assert allocated(DEST)<BATCH_DISK,"batch allocated disk budget exceeded"
    assert allocated(ROOT)<TOTAL_DISK,"total allocated disk budget exceeded"

def guard():
    while not STOP.wait(.2):
        rss=int(next(s.split()[1] for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmRSS:")))*1024
        try:
            if rss>RAM_LIMIT:raise RuntimeError("RSS budget exceeded")
            limits()
        except Exception as exc:
            try:dump(DEST/"resource_limit_abort.json",{"error":repr(exc),"rss_bytes":rss,"originals_retained":True})
            finally:os._exit(90)

class Writer:
    def __init__(self,handle,max_bytes):
        self.f=handle;self.max=max_bytes;self.count=0;self.hash=hashlib.sha256()
    def write(self,data):
        global WRITTEN
        if self.count+len(data)>self.max:raise RuntimeError("package logical write cap")
        if WRITTEN+len(data)>BATCH_DISK:raise RuntimeError("batch logical write cap")
        n=self.f.write(data)
        if n!=len(data):raise IOError("short write")
        self.hash.update(data);self.count+=n;WRITTEN+=n
        return n
    def flush(self):self.f.flush()
    def tell(self):return self.f.tell()

def deny(*args,**kwargs):raise AssertionError("PLY/render/external payload prohibited")

def build_modules(state,cfg,cfgpath,path,reader=None):
    """Actual public builders; legacy PLY allocation is the bounded test hook."""
    def geometry(module,*a,**k):initialize_package_geometry(module,state["model_state_dict"])
    def legacy_load(p,**kwargs):
        assert str(p)==str(path) and kwargs.get("checkpoint_reader") is None
        return state
    def construct():
        if cfg.get("dataset_type")=="scannet":
            model,codec=scan._build_hybrid_model(SimpleNamespace(**cfg),str(path),torch.device("cpu"),checkpoint_reader=reader)
            modules={"model_state_dict":model,"codec_state_dict":codec}
            if "point_summary_adapter_state_dict" in state:
                modules["point_summary_adapter_state_dict"]=scan._build_point_summary_adapter(SimpleNamespace(**cfg),str(path),torch.device("cpu"),checkpoint_reader=reader)
            entry="eval_scannet_pointcloud_radio_gs._build_hybrid_model"
        else:
            model,codec,renderer,sharpener,refiner,config,is_hybrid=ground.load_render_pipeline(str(cfgpath),str(path),torch.device("cpu"),checkpoint_reader=reader,load_ply_rgb_features=False,strict_checkpoint_contract=True)
            assert not config.checkpoint_contract["errors"]
            modules={"model_state_dict":model,"codec_state_dict":codec,"sharpener_state_dict":sharpener}
            if refiner is not None:modules["refiner_state_dict"]=refiner
            if "point_summary_adapter_state_dict" in state:
                modules["point_summary_adapter_state_dict"]=direct._build_point_summary_adapter(config,str(path),torch.device("cpu"),checkpoint_reader=reader)
            entry="eval_lerf_grounding.load_render_pipeline"
            if "point_summary_adapter_state_dict" in state:entry+=" + eval_lerf_direct_3d_selection._build_point_summary_adapter"
        for key,module in modules.items():
            actual=module.state_dict()
            assert set(actual)==set(state[key]),"public module keys differ"
            for name,value in actual.items():
                expected=state[key][name]
                assert value.shape==expected.shape and value.dtype==expected.dtype
                assert tensor_bytes(value)==tensor_bytes(expected),"public module values differ"
            # Archive strides are separately preserved by package comparison.
            # Public load_state_dict uses its existing contiguous layout.
        return modules,entry
    with patch.object(FeatureFieldRenderer,"render_features",deny),patch.object(torch,"load",deny):
        if reader is not None:
            with patch.object(ExplicitFeatureGaussian,"load_from_ply",deny),patch.object(HybridFeatureGaussian,"load_from_ply",deny),patch.object(ground,"resolve_use_2dgs",deny):
                return construct()
        with patch.object(ExplicitFeatureGaussian,"load_from_ply",geometry),patch.object(HybridFeatureGaussian,"load_from_ply",geometry),patch.object(ground,"resolve_use_2dgs",lambda *a:package_use_2dgs(None,state["model_state_dict"])),patch.object(scan,"load_evaluation_checkpoint",legacy_load),patch.object(ground,"load_evaluation_checkpoint",legacy_load),patch.object(direct,"load_evaluation_checkpoint",legacy_load):
            return construct()

def alias_signature(value):
    groups={}
    def walk(x,path):
        if isinstance(x,torch.Tensor):
            storage=x.untyped_storage()
            group=groups.setdefault((storage.data_ptr(),storage.nbytes()),{"capacity_bytes":storage.nbytes(),"views":[]})
            group["views"].append({"path":path,"offset_elements":x.storage_offset(),"dtype":str(x.dtype),"shape":list(x.shape),"stride":list(x.stride())})
        elif isinstance(x,dict):
            for key,v in x.items():walk(v,path+"/"+str(key))
        elif isinstance(x,(list,tuple)):
            for i,v in enumerate(x):walk(v,path+"/"+str(i))
    walk(value,"")
    result=list(groups.values())
    for group in result:group["views"].sort(key=lambda x:x["path"])
    result.sort(key=lambda x:x["views"][0]["path"])
    return result

def source_stable(item):
    assert fingerprint(os.stat(item["source_path"],follow_symlinks=False))==item["fingerprint"],"source identity changed"
    active=processes()
    assert not any(str(Path(item["source_path"]).parent.parent) in p["command"] for p in active),"active writer references source run"

def convert(item,sources):
    global BYTES_READ
    limits();source_stable(item)
    before_raw=item["all_storage_bytes"]-item["optimizer_exclusive_raw_bytes"]
    # 50% allocation reserve plus16MiB per file, based on conservative
    # margin above observed NFS overhead; live/final st_blocks gates also run.
    reserve=int(before_raw*1.5)+16*1024**2
    assert allocated(DEST)+reserve<BATCH_DISK and allocated(ROOT)+reserve<TOTAL_DISK,"insufficient allocation reserve"
    max_new=int(before_raw*1.1)+4*1024**2
    assert BYTES_READ+item["archive_bytes"]+max_new<READ_LIMIT
    cfg=item["config"]
    cfgsource=item["config_source"]
    if not cfgsource.startswith("embedded:"):
        raw=Path(cfgsource).read_bytes();assert json.loads(raw)==cfg,"configuration changed"
        cfgsha=hashlib.sha256(raw).hexdigest()
    else:cfgsha=None
    original,record=read_project_checkpoint_once(item["source_path"],expected_fingerprint=item["fingerprint"])
    BYTES_READ+=record["bytes_read"]
    if cfgsource.startswith("embedded:"):
        assert (original.get("point_summary_adapter_metadata") or {}).get("config_snapshot")==cfg,"embedded configuration changed"
    assert sorted(k for k in original if k.endswith("_state_dict"))==item["state_keys"]
    expected={k:v for k,v in original.items() if k not in DROP_KEYS}
    cfgpath=DEST/(item["id"]+".config.json");dump(cfgpath,cfg)
    old_modules,entry=build_modules(expected,cfg,cfgpath,item["source_path"])
    old_outputs=probes(expected,cfg,loaded_modules=old_modules)
    del old_modules;gc.collect()
    source_stable(item)
    provenance={"source_checkpoint":record,"config_record":{"path":cfgsource,"sha256":cfgsha,"config":cfg},
                "runtime_source_sha256":{x["path"]:x["sha256"] for x in sources},
                "scope":"CPU public-builder parity only; no dataset/teacher/renderer benchmark",
                "recovery_limit":"No exact optimizer/scheduler/scaler resume from this package. Original retained.",
                "external_dependencies":item["external_dependencies"],"admission":item["id"]}
    package=make_inference_package(original,provenance)
    path=DEST/(item["id"]+".inference.pt")
    with open(path,"xb") as f:
        writer=Writer(f,max_new);torch.save(package,writer);writer.flush();os.fsync(f.fileno())
    digest=writer.hash.hexdigest()
    # Drop optimizer/scheduler before loading the new package; expected retains
    # all inference tensors for exact equality and alias-topology comparison.
    del original,package;gc.collect()
    reader=InferencePackageReader(path,expected_sha256=digest)
    loaded=reader.load(path);BYTES_READ+=reader.record["bytes_read"]
    tensors=compare(expected,loaded)
    assert alias_groups(expected)==alias_groups(loaded),"storage alias topology changed"
    storage_signature=alias_signature(expected)
    assert storage_signature==alias_signature(loaded),"storage capacity/offset/alias identity changed"
    assert reader.provenance==provenance
    new_modules,new_entry=build_modules(loaded,cfg,cfgpath,path,reader)
    new_outputs=probes(loaded,cfg,loaded_modules=new_modules)
    output_checks=compare(old_outputs,new_outputs)
    assert entry==new_entry
    source_stable(item)
    # Require two equal allocated observations, not immediate NFS512B results.
    last=None
    for _ in range(30):
        current=path.stat().st_blocks*512
        if current>0 and current==last:break
        last=current;time.sleep(.25)
    else:raise RuntimeError("NFS allocation did not settle")
    limits()
    result={"status":"ACCEPTED_ORIGINAL_RETAINED","id":item["id"],"source":record,
            "source_allocated_bytes":os.stat(item["source_path"]).st_blocks*512,
            "package":reader.record,"package_logical_bytes":writer.count,"package_allocated_bytes":current,
            "conditional_net_allocated_if_original_later_retired":os.stat(item["source_path"]).st_blocks*512-current,
            "config_path":str(cfgpath),"config_sha256":hashlib.sha256(cfgpath.read_bytes()).hexdigest(),
            "entry":entry,"identity":evaluation_package_identity(reader),
            "tensor_count":len(tensors),"tensor_manifest":tensors,"alias_topology_equal":True,
            "storage_capacity_offset_signature":storage_signature,
            "public_output_count":len(output_checks),"public_output_manifest":output_checks,
            "outputs_bitwise_equal":True,"original_stat_unchanged":True,
            "removed_training_keys":sorted(DROP_KEYS.intersection(item["state_keys"])),
            "source_bytes_read":record["bytes_read"],"package_bytes_read":reader.record["bytes_read"],
            "external_dependencies":item["external_dependencies"],"no_overwrite":True,
            "legacy_PLY_test_hook":"geometry allocation from original checkpoint, no PLY read",
            "resume_loss_if_source_retired":"exact optimizer/scheduler/scaler continuation impossible; old checkpoint-path commands must use explicit package flags"}
    dump(DEST/(item["id"]+".acceptance.json"),result)
    print(json.dumps({k:result[k] for k in ["status","id","tensor_count","public_output_count","package_allocated_bytes","conditional_net_allocated_if_original_later_retired"]}),flush=True)
    return {k:v for k,v in result.items() if k not in ("tensor_manifest","public_output_manifest")}

def main():
    global START,DEST
    ap=argparse.ArgumentParser();ap.add_argument("--batch",type=int,choices=(1,2,3),required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
    assert not torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    assert shutil.disk_usage(ROOT).free>20*GIB
    plan=json.loads(PLAN.read_text());pre=json.loads((ROOT/"preflight_v3.json").read_text())
    mapping={x["relative_path"]:x for x in pre["rows"]}
    selected=[mapping[x["relative_path"]] for x in plan["batches"][args.batch-1]]
    assert len(selected)<=8 and len(pre["rows"])==22
    DEST=ROOT/("batch%02d"%args.batch);DEST.mkdir(exist_ok=False)
    START=time.monotonic();sources=runtime_sources(Path.cwd());dump(DEST/"runtime_source_snapshot.json",sources)
    dump(DEST/"batch_plan.json",{"batch":args.batch,"ids":[x["id"] for x in selected],"limits":{"allocated":BATCH_DISK,"total_allocated":TOTAL_DISK,"rss":RAM_LIMIT,"read":READ_LIMIT,"seconds":600}})
    threading.Thread(target=guard,daemon=True).start();results=[]
    try:
        for item in selected:
            if item["admission"]=="SKIP":
                result={"status":"SKIP","id":item["id"],"source_path":item["source_path"],"reason":item["reason"]}
            else:
                try:result=convert(item,sources)
                except Exception as exc:
                    result={"status":"SKIP_AFTER_ADMISSION","id":item["id"],"source_path":item["source_path"],"reason":repr(exc),
                            "original_retained":True,"unaccepted_new_files":[str(p) for p in DEST.glob(item["id"]+".*")]}
                    dump(DEST/(item["id"]+".skip.json"),result);print(json.dumps(result),flush=True)
            results.append(result);gc.collect();limits()
        assert runtime_sources(Path.cwd())==sources,"runtime source changed"
        for item in selected:
            if "fingerprint" in item:source_stable(item)
        summary={"status":"BATCH_COMPLETE_ORIGINALS_RETAINED","batch":args.batch,"results":results,
                 "accepted":sum(x["status"]=="ACCEPTED_ORIGINAL_RETAINED" for x in results),
                 "skipped":sum(x["status"]!="ACCEPTED_ORIGINAL_RETAINED" for x in results),
                 "bytes_read":BYTES_READ,"maxrss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 "allocated_bytes":allocated(DEST),"total_next22_allocated":allocated(ROOT),
                 "elapsed_seconds":time.monotonic()-START,"actual_reclaimed_bytes":0,"deletion_authorized":False}
        assert summary["maxrss_bytes"]<RAM_LIMIT
        dump(DEST/"completion.json",summary)
        print(json.dumps({k:v for k,v in summary.items() if k!="results"}),flush=True)
    finally:STOP.set()
if __name__=="__main__":main()
