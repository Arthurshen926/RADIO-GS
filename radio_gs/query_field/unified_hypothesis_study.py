"""Matched transient-hypothesis controls with a complete five-task barrier."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

from .bounded_experiment import BoundedExperiment, tree_bytes
from .shared_storage import load_field_v2
from .storage import sha256
from .unified_five_benchmark import read, verify_sources, TASKS
from .unified_hypotheses import HypothesisRuntime

MODULE="radio_gs.query_field.unified_hypothesis_study"
RULES=("peak","pooled")


def store_for(config,name,resume=True):
    return BoundedExperiment(config["study_root"],name,config["storage"],resume=resume)


def setup(config):
    root=Path(config["study_root"])
    if not (root / "cohort/closure_complete.json").exists():raise ValueError("complete baseline required")
    plan=read(root / "cohort/plan.json");verify_sources(plan)
    meta=store_for(config,"hypothesis_comparison",False)
    # Existing full baseline is ~6.2GiB. Two prediction/scoring controls add
    # less than 5GiB; do not create even one extra model or optimizer checkpoint.
    meta.preflight(5*2**30)
    sources={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name("unified_hypotheses.py")]}
    design=dict(rules=list(RULES),candidate_count=8,radius_m=.05,margin=.02,
        source_hashes=sources,baseline_plan_sha256=sha256(root / "cohort/plan.json"),
        baseline_summary_sha256=sha256(root / "cohort/summary.json"),
        parameters_inherited="count8 and identity margin0.02 from pre-existing hypothesis policy; radius explicitly normalized to0.05m here (legacy raw-coordinate NMS was not unit-normalized); not swept",
        difference="pooled vs peak ranks the SAME transient member-aware candidate hypotheses",
        baseline_contrast="changes both candidate discovery and candidate composition; not a one-factor contrast to v1",
        no_parameter_update=True,no_new_model_bytes=True,common_rule_all_benchmarks=True,
        explicit_inputs_unchanged=True,category_native_partition_unchanged=True,
        independent_final_test=False,accuracy_promotion=False,all_five_before_feedback=True,
        benchmark_labels_used_for_new_prediction=False)
    meta.json("design.json",design)
    for rule in RULES:
        store=store_for(config,"hypotheses_"+rule,False)
        store.json("lerf_prediction.json",read(root / "cohort/lerf_prediction.json"))
    return design


def verify(config):
    root=Path(config["study_root"]);plan=read(root / "cohort/plan.json");verify_sources(plan)
    design=read(root / "hypothesis_comparison/design.json")
    verify_sources(design)
    if sha256(root / "cohort/plan.json")!=design["baseline_plan_sha256"]:raise ValueError("baseline plan changed")
    return plan,design


def copy_explicit(config,store,job):
    root=Path(config["study_root"])/"cohort/predictions/nvos"/job["name"]/job["name"]
    receipt=read(root/"prediction.json")
    if sha256(root/"predictions.pt")!=receipt["predictions_sha256"]:raise ValueError("explicit baseline changed")
    out=store.root/"predictions/nvos"/job["name"]/job["name"]
    payload=torch.load(root/"predictions.pt",map_location="cpu",weights_only=True)
    store.public_artifact(out/"predictions.pt",payload,"tensor")
    receipt.update(predictions_sha256=sha256(out/"predictions.pt"),
        unchanged_path_reuse=dict(reason="explicit supports immediately dispatch to identical field.query",source=str(root),
                                 source_prediction_sha256=sha256(root/"predictions.pt"),bit_identical_tensors=True))
    store.public_artifact(out/"prediction.json",receipt,"json")


def worker(config,run_name,rule):
    from . import public_lerf, public_scannet, public_uqis
    plan,design=verify(config);store=store_for(config,"hypotheses_"+rule)
    blobs=Path(config["study_root"])/"blobs"
    entry=next(x for x in plan["fields"] if x["run"]==run_name)
    def loader(path,device="cuda",**kwargs):
        field,manifest=load_field_v2(path,device,store_root=blobs)
        return HypothesisRuntime(field,rule,count=design["candidate_count"],radius_m=design["radius_m"],margin=design["margin"]),manifest
    for family,jobs in plan["jobs"].items():
        for job in jobs:
            if job["run"]!=run_name:continue
            out=store.root/"predictions"/family
            if family=="lerf":
                with patch.object(public_lerf,"load_field_versioned",loader):
                    public_lerf.predict(read(store.root/"lerf_prediction.json"),job,out,store_root=blobs,artifact_writer=store)
            elif family=="nvos":copy_explicit(config,store,job)
            elif family=="scannet":
                public_scannet.predict(job["field"],job["queries"],out/job["name"],plan["encoder"],
                                       field_loader=loader,artifact_writer=store)
            else:
                public_uqis.predict(job["field"],Path(plan["uqis_package"])/"method",job["name"],out,plan["encoder"],
                                    field_loader=loader,artifact_writer=store)
    store.json(run_name+"_complete.json",dict(run=run_name,rule=rule,base_field=entry["field"],design_sha256=sha256(Path(config["study_root"])/"hypothesis_comparison/design.json")))


def seal(config):
    plan,design=verify(config);root=Path(config["study_root"]);base=root/"cohort/predictions"
    meta=store_for(config,"hypothesis_comparison");records=[]
    for rule in RULES:
        store=store_for(config,"hypotheses_"+rule);prediction=store.root/"predictions"
        for entry in plan["fields"]:
            if not (store.root/(entry["run"]+"_complete.json")).exists():raise ValueError("missing field control: "+entry["run"])
        for family,jobs in plan["jobs"].items():
            for job in jobs:
                if family=="uqis":
                    receipt=prediction/family/"receipts"/(job["name"]+".json");data=read(receipt)
                    for query in data["queries"]:
                        for folder,suffix,key in [("predictions",".npy","prediction_sha256"),("selections",".npy","selection_sha256"),("gaussians",".pt","gaussian_sha256")]:
                            path=prediction/family/folder/(query["query_id"]+suffix)
                            if sha256(path)!=query[key]:raise ValueError("UQIS control changed")
                            if query["modality"].startswith("point") and suffix==".npy":
                                if not np.array_equal(np.load(path),np.load(base/family/folder/path.name)):
                                    raise ValueError("explicit point output changed")
                else:
                    directory=prediction/family/job["name"];old=base/family/job["name"]
                    if family=="nvos":directory/=job["name"];old/=job["name"]
                    receipt=directory/"prediction.json";data=read(receipt)
                    if sha256(directory/"predictions.pt")!=data["predictions_sha256"]:raise ValueError("control changed")
                    a=torch.load(directory/"predictions.pt",map_location="cpu",weights_only=True)
                    b=torch.load(old/"predictions.pt",map_location="cpu",weights_only=True)
                    if family=="nvos":
                        if set(a)!=set(b) or any(not torch.equal(a[k],b[k]) for k in a):raise ValueError("explicit raster output changed")
                    elif family=="scannet":
                        if not torch.equal(a["native"],b["native"]):raise ValueError("native category output changed")
                    elif not torch.equal(a["scores"]["native_generic_relevance"],b["scores"]["native_generic_relevance"]):
                        raise ValueError("native text function changed")
                if data["field_sha256"]!=read(Path(job["field"])/"manifest.json")["state_identity"]:raise ValueError("field changed")
                records.append(dict(rule=rule,family=family,scene=job["name"],receipt=str(receipt),sha256=sha256(receipt)))
        expected={q for ids in plan["uqis_query_ids"].values() for q in ids}
        for folder in ["predictions","selections","gaussians"]:
            if {p.stem for p in (prediction/"uqis"/folder).iterdir()}!=expected:raise ValueError("incomplete UQIS denominator")
    result=dict(tasks=list(TASKS),rules=list(RULES),records=records,explicit_points_and_rasters_bit_identical=True,
        native_functions_bit_identical=True,all_prediction_seal_before_new_scoring=True)
    path=meta.root/"prediction_barrier.json"
    if path.exists():
        if read(path)!=result:raise ValueError("barrier changed")
    else:meta.json("prediction_barrier.json",result)
    return result


def score(config):
    seal(config);plan,_=verify(config)
    from . import public_lerf, public_nvos, public_scannet
    for rule in RULES:
        store=store_for(config,"hypotheses_"+rule);out=store.root/"predictions"
        for job in plan["jobs"]["lerf"]:
            if not (out/"lerf"/job["name"]/"evaluation/evaluation.json").exists():
                public_lerf.score(read(store.root/"lerf_prediction.json"),job,out/"lerf")
        for job in plan["jobs"]["nvos"]:
            if not (out/"nvos"/job["name"]/"evaluation.json").exists():public_nvos.score(job["scoring"],out/"nvos"/job["name"])
        for job in plan["jobs"]["scannet"]:
            directory=out/"scannet"/job["name"]
            if not (directory/"evaluation.json").exists():public_scannet.score(directory,job["evaluation"],directory/"evaluation.json")
        if not (out/"uqis/evaluation.json").exists():
            subprocess.run([sys.executable,"-m","radio_gs.benchmarks.uqis.evaluate","--package",plan["uqis_package"],
                "--predictions",str(out/"uqis/predictions"),"--selections",str(out/"uqis/selections"),"--output",str(out/"uqis/evaluation.json")],check=True)
        store.json("scoring_complete.json",dict(tasks=list(TASKS),rule=rule,full_cohort=True))


def run_all(config_path,gpus):
    config=read(config_path);plan,_=verify(config);meta=store_for(config,"hypothesis_comparison")
    def lane(rule,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",PYTHONPATH=".")
        store=store_for(config,"hypotheses_"+rule)
        for entry in plan["fields"]:
            name=entry["run"]
            if (store.root/(name+"_complete.json")).exists():continue
            print(rule,name,"predict",flush=True)
            with (store.root/(name+".log")).open("x") as handle:
                subprocess.run([config["predict_python"],"-m",MODULE,"worker","--config",config_path,"--rule",rule,"--run",name],
                               env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        return rule
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(lane,rule,gpus[i%len(gpus)]) for i,rule in enumerate(RULES)]
        for future in futures:future.result()
    seal(config)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[0]),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",PYTHONPATH=".")
    with (meta.root/"scoring.log").open("x") as handle:
        subprocess.run([config["predict_python"],"-m",MODULE,"score","--config",config_path],env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    print("all five controls scored",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=["setup","worker","run","seal","score"])
    p.add_argument("--config",default="configs/query_field/unified_five_benchmark.json");p.add_argument("--rule",choices=RULES)
    p.add_argument("--run");p.add_argument("--gpus",nargs="+",type=int,default=[0,5]);a=p.parse_args()
    config=read(a.config);torch.set_num_threads(4)
    if a.stage in {"worker","score"}:torch.cuda.set_per_process_memory_fraction(.45)
    if a.stage=="setup":result=setup(config)
    elif a.stage=="worker":result=worker(config,a.run,a.rule)
    elif a.stage=="run":result=run_all(a.config,a.gpus)
    elif a.stage=="seal":result=seal(config)
    else:result=score(config)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
