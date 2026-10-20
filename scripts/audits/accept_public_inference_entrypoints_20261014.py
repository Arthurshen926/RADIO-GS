"""Bounded acceptance of the three inference packages through public builders."""
from __future__ import annotations
import builtins, gc, hashlib, json, os, resource, shutil, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from radio_gs.utils.evaluation_checkpoint import InferencePackageReader, initialize_package_geometry, package_use_2dgs
from radio_gs.utils.inference_checkpoint import fingerprint, read_project_checkpoint_once
from radio_gs.scripts import eval_scannet_pointcloud_radio_gs as scan
from radio_gs.scripts import eval_lerf_grounding as grounding
from radio_gs.scripts import eval_lerf_direct_3d_selection as direct
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from scripts.audits.inference_pilot_probes import probes
from scripts.audits.pilot_inference_checkpoint_20261014 import tensor_bytes, runtime_sources

ROOT=Path("/mnt/pool/sqy/results/RADIO-GS/output/storage_inference_pilot_20261014_1146")
DEST=ROOT/"public_entry_acceptance_1223"
LIMIT=3*1024**3

def dump(path,value):
    with open(path,"x") as f:json.dump(value,f,indent=2)

def deny(*a,**k):
    raise AssertionError("External payload/PLY/render access is prohibited in CPU acceptance")

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available()
    assert shutil.disk_usage(ROOT).free>20*1024**3
    receipt=json.loads((ROOT/"final_acceptance.json").read_text())
    DEST.mkdir(exist_ok=False)
    sources=runtime_sources(Path.cwd())
    dump(DEST/"runtime_source_snapshot.json",sources)
    initial={x["source_path"]:fingerprint(os.stat(x["source_path"],follow_symlinks=False)) for x in receipt["packages"]}
    for x in receipt["packages"]:assert initial[x["source_path"]]==x["source_fingerprint"]
    stop=threading.Event()
    def guard():
        while not stop.wait(.05):
            rss=int(next(s.split()[1] for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmRSS:")))*1024
            if rss>LIMIT:os._exit(90)
    threading.Thread(target=guard,daemon=True).start()
    results=[];started=time.monotonic()
    try:
        for item in receipt["packages"]:
            reader=InferencePackageReader(item["package_path"],expected_sha256=item["package_sha256"])
            ckpt=reader.load(item["package_path"])
            original=json.loads(Path(item["acceptance_path"]).read_text())
            cfg=reader.provenance["config_record"]["config"]
            cfgpath=DEST/(item["name"]+".config.json")
            dump(cfgpath,cfg)
            actual_modules={}
            with patch.object(ExplicitFeatureGaussian,"load_from_ply",deny), patch.object(HybridFeatureGaussian,"load_from_ply",deny), patch.object(grounding,"resolve_use_2dgs",deny), patch.object(FeatureFieldRenderer,"render_features",deny):
                if item["name"]=="scannet_hybrid":
                    model,codec=scan._build_hybrid_model(SimpleNamespace(**cfg),item["package_path"],torch.device("cpu"),checkpoint_reader=reader)
                    actual_modules={"model_state_dict":model,"codec_state_dict":codec}
                    entry="eval_scannet_pointcloud_radio_gs._build_hybrid_model"
                else:
                    build=grounding.load_render_pipeline if item["name"]=="lerf_explicit" else direct.load_render_pipeline
                    model,codec,renderer,sharpener,refiner,loaded_cfg,is_hybrid=build(str(cfgpath),item["package_path"],torch.device("cpu"),checkpoint_reader=reader,load_ply_rgb_features=False)
                    assert not loaded_cfg.checkpoint_contract["errors"]
                    actual_modules={"model_state_dict":model,"codec_state_dict":codec,"sharpener_state_dict":sharpener}
                    if refiner is not None:actual_modules["refiner_state_dict"]=refiner
                    if item["name"]=="vpr_adapter":
                        adapter=direct._build_point_summary_adapter(loaded_cfg,item["package_path"],torch.device("cpu"),checkpoint_reader=reader)
                        actual_modules["point_summary_adapter_state_dict"]=adapter
                    entry="eval_lerf_grounding.load_render_pipeline" if item["name"]=="lerf_explicit" else "eval_lerf_direct_3d_selection.load_render_pipeline + _build_point_summary_adapter"
                # Hash actual constructed state, not a reconstructed replacement.
                expected={r["path"]:r for r in original["tensor_manifest"]}
                checked=0
                for key,module in actual_modules.items():
                    for name,value in module.state_dict().items():
                        row=expected["/"+key+"/"+name]
                        assert list(value.shape)==row["shape"] and str(value.dtype)==row["dtype"]
                        assert hashlib.sha256(tensor_bytes(value)).hexdigest()==row["sha256"]
                        checked+=1
                outputs=probes(ckpt,cfg,loaded_modules=actual_modules)
                baseline={r["path"]:r for r in original["probes"]}
                assert set("/"+k for k in outputs)==set(baseline)
                # Same public builders, legacy checkpoint branch, identical config.
                # Only PLY allocation is substituted with the *original* frozen
                # checkpoint geometry; rendering/datasets are outside this test.
                source_state, source_record = read_project_checkpoint_once(
                    item["source_path"], expected_sha256=item["source_sha256"],
                    expected_fingerprint=item["source_fingerprint"])
                def legacy_load(path, **kwargs):
                    assert str(path)==item["source_path"] and kwargs.get("checkpoint_reader") is None
                    return source_state
                def seed_legacy_geometry(module, *args, **kwargs):
                    initialize_package_geometry(module, source_state["model_state_dict"])
                with patch.object(ExplicitFeatureGaussian,"load_from_ply",seed_legacy_geometry), patch.object(HybridFeatureGaussian,"load_from_ply",seed_legacy_geometry), patch.object(grounding,"resolve_use_2dgs",lambda *a: package_use_2dgs(None,source_state["model_state_dict"])), patch.object(scan,"load_evaluation_checkpoint",legacy_load), patch.object(grounding,"load_evaluation_checkpoint",legacy_load), patch.object(direct,"load_evaluation_checkpoint",legacy_load):
                    if item["name"]=="scannet_hybrid":
                        old_model,old_codec=scan._build_hybrid_model(SimpleNamespace(**cfg),item["source_path"],torch.device("cpu"))
                        old_modules={"model_state_dict":old_model,"codec_state_dict":old_codec}
                    else:
                        old_model,old_codec,old_renderer,old_sharpener,old_refiner,old_cfg,old_hybrid=build(str(cfgpath),item["source_path"],torch.device("cpu"),load_ply_rgb_features=False,strict_checkpoint_contract=True)
                        old_modules={"model_state_dict":old_model,"codec_state_dict":old_codec,"sharpener_state_dict":old_sharpener}
                        if old_refiner is not None:old_modules["refiner_state_dict"]=old_refiner
                        if item["name"]=="vpr_adapter":
                            old_adapter=direct._build_point_summary_adapter(old_cfg,item["source_path"],torch.device("cpu"))
                            old_modules["point_summary_adapter_state_dict"]=old_adapter
                legacy_outputs=probes(source_state,cfg,loaded_modules=old_modules)
                reference=probes(source_state,cfg)
                for k,v in reference.items():assert hashlib.sha256(tensor_bytes(v)).hexdigest()==baseline["/"+k]["sha256"],"original frozen probe changed: "+k
                for k,v in outputs.items():assert tensor_bytes(v)==tensor_bytes(legacy_outputs[k]),"public legacy/package mismatch: "+k
                strides=[]
                for key,module in actual_modules.items():
                    for name,v in module.state_dict().items():
                        if tuple(v.stride())!=tuple(ckpt[key][name].stride()):
                            strides.append({"key":key+"/"+name,"public_stride":list(v.stride()),"archive_stride":list(ckpt[key][name].stride())})
                probe_layout_differences=[{"name":k,"max_abs":float((v-reference[k]).abs().max()),"relative_linf":float((v-reference[k]).abs().max())/max(float(reference[k].abs().max()),1e-30)} for k,v in outputs.items() if tensor_bytes(v)!=tensor_bytes(reference[k])]
                assert reader.load(item["package_path"]) is ckpt
            row={"name":item["name"],"entry":entry,"package_path":item["package_path"],"package_sha256":reader.record["sha256"],"source_checkpoint_sha256":item["source_sha256"],"package_bytes_read":reader.record["bytes_read"],"public_module_tensor_count":checked,"all_public_tensors_equal_to_frozen_original":True,"numerical_outputs_equal_to_frozen_original":len(outputs),"external_PLY_teacher_render_reads":0,"config_snapshot":str(cfgpath)}
            row.update({"source_bytes_read":source_record["bytes_read"],"legacy_public_outputs_bitwise_equal":len(outputs),"archival_probe_exactly_reproduced":len(reference),"public_vs_archival_layout_differences":strides,"public_vs_archival_probe_differences":probe_layout_differences,"legacy_PLY_allocation_test_hook":"original checkpoint geometry, no PLY read"})
            dump(DEST/(item["name"]+".json"),row);results.append(row);print(json.dumps(row),flush=True)
            del ckpt,reader,model,codec,actual_modules,outputs,source_state,old_model,old_codec,old_modules,legacy_outputs,reference
            if item["name"]!="scannet_hybrid":del old_renderer,old_sharpener,old_refiner,old_cfg
            if item["name"]=="vpr_adapter":del old_adapter
            if item["name"]!="scannet_hybrid":del renderer,sharpener,refiner,loaded_cfg
            if item["name"]=="vpr_adapter":del adapter
            gc.collect()
        assert runtime_sources(Path.cwd())==sources
        for p,f in initial.items():assert fingerprint(os.stat(p,follow_symlinks=False))==f
        usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        assert usage<LIMIT
        summary={"status":"PUBLIC_BUILDERS_CPU_ACCEPTED","scope":"Real public evaluation loader/builders plus deterministic CPU module probes; not full CLI dataset/render/GPU benchmark","packages":results,"package_bytes_read":sum(x["package_bytes_read"] for x in results),"original_payload_bytes_read":sum(x["source_bytes_read"] for x in results),"maxrss_bytes":usage,"elapsed_seconds":time.monotonic()-started,"source_fingerprints_unchanged":True,"originals_retained":True,"actual_reclaimed_bytes":0,"source_snapshot":str(DEST/"runtime_source_snapshot.json")}
        dump(DEST/"completion.json",summary);print(json.dumps(summary),flush=True)
    except Exception as exc:
        dump(DEST/"failure.json",{"status":"FAILED_ORIGINALS_UNCHANGED","error":repr(exc),"accepted_before_failure":results})
        raise
    finally:stop.set()

if __name__=="__main__":main()
