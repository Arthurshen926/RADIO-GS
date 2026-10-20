"""Read-only admission metadata for the exact authorized 22 checkpoints."""
import collections,io,json,os,pickle,stat,time,zipfile
from pathlib import Path
import numpy as np

ROOT=Path("/mnt/pool/sqy/results/RADIO-GS/output/radio_gs")
PLAN=Path(__file__).with_name("next22_inference_plan_20261014.json")
DEST=ROOT.parent/"storage_inference_next22_20261014_1230"

def tensor(storage,offset,size,stride,*args):
    return {"__tensor__":True,"storage":storage,"offset":offset,"shape":list(size),"stride":list(stride)}
def parameter(value,*args):return value
class Metadata(pickle.Unpickler):
    def persistent_load(self,pid):
        assert pid[0]=="storage"
        return {"kind":pid[1],"id":pid[2],"location":pid[3],"count":pid[4]}
    def find_class(self,module,name):
        if module=="_codecs" and name=="encode":
            def latin1(value,encoding):
                if encoding not in ("latin1","latin-1"):raise ValueError("unapproved bytes encoding")
                return value.encode("latin1")
            return latin1
        if module=="collections" and name=="OrderedDict":return collections.OrderedDict
        if module=="torch" and name.endswith("Storage"):return name
        if module=="torch._utils" and name in ("_rebuild_tensor","_rebuild_tensor_v2"):return tensor
        if module=="torch._utils" and name in ("_rebuild_parameter","_rebuild_parameter_with_state"):return parameter
        if module=="numpy" and name=="dtype":return np.dtype
        if module in ("numpy.core.multiarray","numpy._core.multiarray") and name=="scalar":return np.core.multiarray.scalar
        raise ValueError("unknown pickle metadata global "+module+"."+name)

def fp(path):
    s=os.stat(path,follow_symlinks=False)
    return {k:int(getattr(s,"st_"+k)) for k in ("dev","ino","mode","nlink","size","mtime_ns","ctime_ns")}

def processes():
    out=[]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=(p/"cmdline").read_bytes().replace(b"\x00",b" ").decode(errors="replace")
            if "python" in cmd and ("train" in cmd or "radio_gs" in cmd) and "preflight_next22" not in cmd and "webcodex" not in cmd:
                out.append({"pid":int(p.name),"command":cmd[:2000]})
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return out

def main():
    plan=json.loads(PLAN.read_text());DEST.mkdir(exist_ok=True)
    active=processes();rows=[];read_bytes=0
    for n,row in enumerate(plan["rows"]):
        path=ROOT/row["relative_path"];item=dict(row);item["source_path"]=str(path);item["id"]="n%02d"%(n+1)
        try:
            before=fp(path)
            assert stat.S_ISREG(before["mode"]) and before["nlink"]==1 and before["size"]==row["archive_bytes"]
            assert before["size"]<=800_000_000
            with zipfile.ZipFile(path) as z:
                names=[x for x in z.namelist() if x.endswith("/data.pkl")]
                assert len(names)==1
                assert z.getinfo(names[0]).file_size<2*1024**2
                raw=z.read(names[0]);read_bytes+=len(raw)
            meta=Metadata(io.BytesIO(raw)).load()
            assert fp(path)==before
            cfgpath=path.parent.parent/"reports/resolved_config.json"
            if cfgpath.exists():
                rawcfg=cfgpath.read_bytes();cfg=json.loads(rawcfg);cfgsource=str(cfgpath)
            else:
                cfg=(meta.get("point_summary_adapter_metadata") or {}).get("config_snapshot")
                cfgsource="embedded:point_summary_adapter_metadata.config_snapshot"
            assert isinstance(cfg,dict),"missing resolved configuration"
            item["fingerprint"]=before;item["config"]=cfg;item["config_source"]=cfgsource
            item["state_keys"]=sorted(k for k in meta if k.endswith("_state_dict"))
            item["architecture"]=cfg.get("architecture","explicit");item["codec_type"]=cfg.get("codec_type","hcd");item["dataset_type"]=cfg.get("dataset_type","")
            metadata=meta.get("point_summary_adapter_metadata") or {};contract=metadata.get("direct_head_contract") or {}
            item["adapter_context"]=contract.get("point_summary_adapter_context_features",metadata.get("point_summary_adapter_context_features",cfg.get("point_summary_adapter_context_features","")))
            item["adapter_metadata"]=metadata
            external={k:v for k,v in cfg.items() if isinstance(v,str) and v and any(w in k for w in ("cache","path","root","checkpoint","weights","feature_dir"))}
            item["external_dependencies"]={k:{"path":v,"exists":Path(v).exists() if not ("{" in v or "}" in v) else None,"classification":"input/provenance reference; runtime use depends on evaluation flags, not removed"} for k,v in external.items()}
            supported={"model_state_dict","codec_state_dict","sharpener_state_dict","refiner_state_dict","depth_head_state_dict","seg_head_state_dict","point_summary_adapter_state_dict","optimizer_state_dict","scheduler_state_dict","scaler_state_dict"}
            unknown=set(item["state_keys"])-supported
            if unknown:raise ValueError("unknown state schema: "+repr(unknown))
            if item["architecture"] not in ("hybrid","explicit"):raise ValueError("unsupported architecture")
            if item["codec_type"] not in ("hcd","direct"):raise ValueError("unadmitted codec "+item["codec_type"])
            if item["adapter_context"]:raise ValueError("nonempty adapter context requires separately admitted public forward path")
            if item["dataset_type"] not in ("scannet","lerf"):raise ValueError("dataset has no admitted task-specific public CLI: "+item["dataset_type"])
            if any(str(path.parent.parent) in p["command"] or str(path) in p["command"] for p in active):raise ValueError("visible active process references source run")
            item["admission"]="ADMIT_TO_CPU_CONVERSION"
        except Exception as exc:item["admission"]="SKIP";item["reason"]=repr(exc)
        rows.append(item)
    out={"status":"METADATA_PREFLIGHT_ONLY","created_epoch":time.time(),"rows":rows,"metadata_bytes_read":read_bytes,"visible_candidate_processes":active,"writer_limit":"Container-local process visibility only; remote writers cannot be excluded. Frozen descriptor/source stats and final hashes are mandatory."}
    with open(DEST/"preflight_v3.json","x") as f:json.dump(out,f,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
    print(json.dumps({"output":str(DEST/"preflight_v3.json"),"metadata_bytes_read":read_bytes,"rows":[{k:r.get(k) for k in ["id","relative_path","admission","reason","architecture","codec_type","dataset_type","adapter_context"]} for r in rows],"visible_candidate_process_count":len(active)}))
if __name__=="__main__":main()
