"""One compact multi-teacher recipe, complete five-task coverage, then feedback.

Training never opens task queries/labels. All task predictions must be sealed
before the scorer can load any target annotations. Failed or unsupported queries
cannot disappear from the declared denominator. No task-specific winner routing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .bounded_experiment import BoundedExperiment, tree_bytes
from .storage import sha256

FAMILIES = ("lerf", "nvos", "scannet", "uqis")
TASKS = ("lerf2d", "lerf3d_local_projection", "nvos", "scannet_ovs", "uqis")
MODULE = "radio_gs.query_field.unified_five_benchmark"


def read(path):
    return json.loads(Path(path).read_text())


def control(config, resume=True):
    return BoundedExperiment(config["study_root"], "cohort", config["storage"], resume=resume)


def prepare(config):
    import torch
    old = read(config["cohort"])
    recipe = read(config["training_recipe"])
    if config["architecture"] != "unified_compact_v1" or config["arm"] != "functional":
        raise ValueError("this cohort is frozen to one complete multi-teacher baseline")
    if config["all_teachers"] != ["SigLIP2", "DINOv2", "SAM"] or tuple(config["benchmark_gate"]) != TASKS:
        raise ValueError("all teachers and all five tasks are required")
    recipe.update(storage=config["storage"], audit_teacher_gradients=True)
    if min(recipe["functional_weight"], recipe["membership_weight"]) <= 0:
        raise ValueError("all teacher objectives must be active")
    root = Path(config["study_root"])
    jobs, fields, seen = {f: [] for f in FAMILIES}, [], {}
    for family in FAMILIES:
        for old_job in old[family]:
            name = old_job.get("scene", old_job.get("name"))
            source = (Path(old_job["field"]).parents[1] / "observations").resolve(strict=True)
            if str(source) not in seen:
                manifest = read(source / "observations.json")
                if (manifest.get("benchmark_queries_used") is not False or
                        manifest.get("semantic_labels_used") is not False or
                        manifest.get("feature_development_excluded") is not True):
                    raise ValueError("source legality not established: " + name)
                if not manifest.get("dino_checkpoint") or not manifest.get("native_semantic_space"):
                    raise ValueError("native semantic/local teacher provenance missing")
                for frame in manifest["frames"]:
                    if not frame.get("sam_checkpoint") or not frame.get("masks", 0):
                        raise ValueError("membership teacher missing: " + name)
                    if not (source / f'view_{frame["frame"]:05d}.pt').is_file():
                        raise ValueError("compiled multi-teacher view missing")
                initial = torch.load(source / "initial.pt", map_location="cpu", weights_only=True, mmap=True)
                n = len(initial["observed"])
                if (initial["semantic"].shape != (n,1536) or initial["local"].shape != (n,768)):
                    raise ValueError("native teacher dimension differs")
                run = family + "_" + name
                # Same size formula for every scene; changes only storage
                # reservations, never latent width, steps or sampling recipe.
                state_bytes = n * (4 * recipe["latent_dim"] + 57) + 2 ** 21
                checkpoint = state_bytes + 2 * n * 4 * recipe["latent_dim"] + 2 ** 24
                export = state_bytes + 2 ** 20
                cfg = dict(recipe, observations=str(source), checkpoint_budget_bytes=checkpoint,
                           export_budget_bytes=export)
                fields.append(dict(run=run, family=family, scene=name, elements=n,
                    observed=int(initial["observed"].sum()), observations=str(source),
                    observations_sha256=sha256(source / "observations.json"),
                    initial_sha256=manifest["initial"]["sha256"], config=cfg,
                    field=str(root / run / "field"), train_views=len(manifest["splits"]["train_frames"]),
                    development_views=len(manifest["splits"]["development_frames"]), all_teachers=True,
                    estimated_state_bytes=state_bytes, peak_reservation_bytes=3*checkpoint+export))
                seen[str(source)] = fields[-1]
                del initial
            entry = seen[str(source)]
            job = dict(name=name, field=entry["field"], run=entry["run"])
            if family == "lerf":
                job.update(scene=name, queries=f"configs/consolidated/{name}.queries.json",
                           evaluation_scene=f"output/system_map_rebuild/{name}/complete/scene.json")
            elif family == "nvos":
                pred = read(old_job["prediction_config"])
                pred["jobs"][0].update(field=entry["field"], query_program="parent")
                score = read(old_job["scoring_config"])
                score["prediction_config"] = str(root / "cohort" / ("nvos_" + name + "_prediction.json"))
                job.update(prediction=pred, scoring=score)
            elif family == "scannet":
                job.update(queries=old_job["queries"], evaluation=old_job["evaluation"])
            jobs[family].append(job)
    # Interleave families from the start; ScanNet cannot occupy the whole queue.
    grouped = {f:[x for x in fields if x["family"] == f] for f in FAMILIES}
    fields = [grouped[f][i] for i in range(max(map(len, grouped.values()))) for f in FAMILIES if i < len(grouped[f])]
    public = Path(old["uqis_package"]) / "method"
    qids = {m:[q["query_id"] for q in read(public / "queries" / (m + ".json"))["queries"]]
            for m in ("text", "image", "point_2d", "point_3d")}
    expected = dict(lerf_scenes=len(jobs["lerf"]), lerf_queries=sum(len(read(j["queries"])["queries"]) for j in jobs["lerf"]),
                    nvos_cases=len(jobs["nvos"]), scannet_rooms=len(jobs["scannet"]), uqis_rooms=len(jobs["uqis"]),
                    uqis_queries=sum(map(len,qids.values())), uqis_queries_by_modality={m:len(v) for m,v in qids.items()})
    if (expected["lerf_scenes"], expected["nvos_cases"], expected["scannet_rooms"], expected["uqis_rooms"], expected["uqis_queries"]) != (4,8,10,9,268):
        raise ValueError("full local cohort changed")
    estimate = sum(f["estimated_state_bytes"] for f in fields)
    # Two GiB reserved for bounded prediction payloads, scorer evidence and logs.
    peak = estimate + max(f["peak_reservation_bytes"] for f in fields) + 2 * 2 ** 30
    if peak > config["storage"]["study_bytes"]:
        raise ValueError("full cohort cannot fit declared study policy")
    store = control(config, resume=False)
    store.preflight(min(peak, config["storage"]["run_bytes"]))
    sources = {str(p):sha256(p) for p in Path("radio_gs/query_field").glob("*.py")}
    plan = dict(schema="radio_gs.unified_five_plan.v1", config=config, fields=fields, jobs=jobs,
        expected=expected, uqis_query_ids=qids, uqis_package=old["uqis_package"], encoder=old["encoder"],
        source_hashes=sources, shared_recipe=recipe, per_scene_weights=True, shared_cross_scene_weights=False,
        complete_multimode_transfer=False, estimated_retained_model_bytes=estimate,
        conservative_peak_bytes=peak, all_predictions_before_scoring=True,
        benchmark_specific_routing=False, independent_final_test=False)
    store.json("plan.json", plan)
    for job in jobs["nvos"]:
        store.json("nvos_" + job["name"] + "_prediction.json", job["prediction"])
    store.json("lerf_prediction.json", dict(shared_encoder=old["encoder"],
        generic_negative_texts=["object", "things", "stuff", "texture"], native_temperature=10.,
        jobs=jobs["lerf"], architecture=config["architecture"], threshold=.5))
    return dict(expected=expected, fields=len(fields), retained_model_gib=estimate/2**30, peak_gib=peak/2**30)


def verify_sources(plan):
    changed = [p for p,h in plan["source_hashes"].items() if sha256(p) != h]
    if changed:
        raise ValueError("frozen method source changed: " + str(changed))


def worker(config, run_name, phase):
    import torch
    store = control(config)
    plan = read(store.root / "plan.json")
    verify_sources(plan)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"])
    entry = next(f for f in plan["fields"] if f["run"] == run_name)
    if phase == "train":
        from .unified_training import run
        return run(entry["config"], config["arm"], config["study_root"], device="cuda", run_name=run_name)
    from . import public_lerf, public_nvos, public_scannet, public_uqis
    root = store.root / "predictions"
    blobs = Path(config["study_root"]) / "blobs"
    for family in FAMILIES:
        for job in plan["jobs"][family]:
            if job["run"] != run_name:
                continue
            if family == "lerf":
                public_lerf.predict(read(store.root / "lerf_prediction.json"), job, root / family,
                                    store_root=blobs, artifact_writer=store)
            elif family == "nvos":
                pred = job["prediction"]
                public_nvos.predict(pred, pred["jobs"][0], root / family / job["name"],
                                    store_root=blobs, artifact_writer=store)
            elif family == "scannet":
                public_scannet.predict(job["field"],job["queries"],root / family / job["name"],plan["encoder"],
                                       store_root=blobs, artifact_writer=store)
            else:
                public_uqis.predict(job["field"],Path(plan["uqis_package"]) / "method",job["name"],root / family,
                                    plan["encoder"],store_root=blobs, artifact_writer=store)
    return dict(run=run_name, prediction_complete=True)


def seal(config):
    store = control(config)
    plan = read(store.root / "plan.json")
    verify_sources(plan)
    root = store.root / "predictions"
    records = []
    for entry in plan["fields"]:
        run = Path(config["study_root"]) / entry["run"]
        trained = read(run / "training.json")
        audit = read(run / "teacher_gradient_audit.json")
        if not audit["all_teachers_active"] or set(audit["teachers"]) != {"semantic","local","membership"}:
            raise ValueError("all-teacher gradient audit incomplete")
        if read(run / "design.json")["config"] != entry["config"]:
            raise ValueError("scene-specific recipe change")
        if read(run / "field/manifest.json")["state_identity"] != trained["state_identity"]:
            raise ValueError("field identity changed")
    for family in FAMILIES:
        for job in plan["jobs"][family]:
            if family == "uqis":
                receipt = root / family / "receipts" / (job["name"] + ".json")
                data = read(receipt)
                if data["field_spec"]["architecture"] != config["architecture"]:
                    raise ValueError("mixed field architectures")
                for q in data["queries"]:
                    for folder, suffix, key in [("predictions",".npy","prediction_sha256"),
                            ("selections",".npy","selection_sha256"),("gaussians",".pt","gaussian_sha256")]:
                        p = root / family / folder / (q["query_id"] + suffix)
                        if sha256(p) != q[key]:
                            raise ValueError("UQIS prediction changed")
            else:
                directory = root / family / job["name"]
                if family == "nvos":
                    directory /= job["name"]
                receipt = directory / "prediction.json"
                data = read(receipt)
                if sha256(directory / "predictions.pt") != data["predictions_sha256"]:
                    raise ValueError("prediction changed")
            manifest = read(Path(job["field"]) / "manifest.json")
            if data["field_sha256"] != manifest["state_identity"]:
                raise ValueError("predictions bound to a different field")
            if family == "lerf" and data["queries"] != len(read(job["queries"])["queries"]):
                raise ValueError("missing LERF query")
            if family == "scannet" and len(data["queries"]) != len(read(job["queries"])["queries"]):
                raise ValueError("missing category")
            records.append(dict(family=family,scene=job["name"],receipt=str(receipt),sha256=sha256(receipt)))
    expected = {q for ids in plan["uqis_query_ids"].values() for q in ids}
    for folder in ("predictions", "selections", "gaussians"):
        if {p.stem for p in (root / "uqis" / folder).iterdir()} != expected:
            raise ValueError("UQIS full denominator incomplete")
    result = dict(plan_sha256=sha256(store.root / "plan.json"), records=records,
                  tasks=list(TASKS), expected=plan["expected"], all_teachers=True)
    barrier = store.root / "prediction_barrier.json"
    if barrier.exists():
        if read(barrier) != result:
            raise ValueError("sealed cohort changed")
    else:
        store.json("prediction_barrier.json", result)
    return result


def score(config):
    # This is the only path to annotation readers. Re-verify all five families.
    seal(config)
    store = control(config); plan = read(store.root / "plan.json")
    root = store.root / "predictions"
    from . import public_lerf, public_nvos, public_scannet
    for job in plan["jobs"]["lerf"]:
        if not (root / "lerf" / job["name"] / "evaluation/evaluation.json").exists():
            public_lerf.score(read(store.root / "lerf_prediction.json"),job,root / "lerf")
    for job in plan["jobs"]["nvos"]:
        if not (root / "nvos" / job["name"] / "evaluation.json").exists():
            public_nvos.score(job["scoring"],root / "nvos" / job["name"])
    for job in plan["jobs"]["scannet"]:
        out = root / "scannet" / job["name"]
        if not (out / "evaluation.json").exists():
            public_scannet.score(out,job["evaluation"],out / "evaluation.json")
    if not (root / "uqis/evaluation.json").exists():
        subprocess.run([sys.executable,"-m","radio_gs.benchmarks.uqis.evaluate","--package",plan["uqis_package"],
            "--predictions",str(root / "uqis/predictions"),"--selections",str(root / "uqis/selections"),
            "--output",str(root / "uqis/evaluation.json")],check=True)
    store.json("scoring_complete.json", dict(tasks=list(TASKS),plan_sha256=sha256(store.root / "plan.json"),
        all_teachers=True, official_equivalence=False, independent_final_test=False))
    return dict(five_benchmarks_complete=True)


def run_all(config_path, gpu):
    config = read(config_path); store = control(config); plan = read(store.root / "plan.json")
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",PYTHONPATH=".")
    completed, failures = [], []
    for entry in plan["fields"]:
        name = entry["run"]
        for phase in ("train", "predict"):
            done = ((Path(config["study_root"]) / name / "training.json").exists() if phase == "train"
                    else (store.root / (name + "_prediction_complete.json")).exists())
            if done:
                continue
            log = store.root / f"{name}_{phase}.log"
            if log.exists():
                failures.append(dict(run=name,phase=phase,error="unfinished prior attempt; inspect before retry"))
                break
            print(json.dumps(dict(run=name,phase=phase,status="started")),flush=True)
            with log.open("x") as handle:
                proc = subprocess.run([config["train_python" if phase == "train" else "predict_python"],
                    "-m",MODULE,phase,"--config",str(config_path),"--run",name],env=env,stdout=handle,stderr=subprocess.STDOUT)
            if proc.returncode:
                failures.append(dict(run=name,phase=phase,exit_code=proc.returncode,log=str(log)))
                print(json.dumps(failures[-1]),flush=True)
                break
            if phase == "predict":
                store.json(name + "_prediction_complete.json",dict(run=name,plan_sha256=sha256(store.root / "plan.json")))
                completed.append(name)
        store.json("progress.json",dict(completed=completed,failures=failures,study_bytes=tree_bytes(store.study),
                   total_fields=len(plan["fields"]),complete=False),replace=True)
    if failures:
        raise RuntimeError("Full-cohort failures retained; no partial-cohort score/promotion: " + str(failures))
    seal(config)
    with (store.root / "scoring.log").open("x") as handle:
        subprocess.run([config["predict_python"],"-m",MODULE,"score","--config",str(config_path)],
                       env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    return dict(five_benchmarks_complete=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","run","train","predict","seal","score"])
    p.add_argument("--config",default="configs/query_field/unified_five_benchmark.json")
    p.add_argument("--gpu",type=int,default=4);p.add_argument("--run")
    args = p.parse_args(); config = read(args.config)
    if args.stage == "prepare": result=prepare(config)
    elif args.stage == "run": result=run_all(args.config,args.gpu)
    elif args.stage in ("train","predict"): result=worker(config,args.run,args.stage)
    elif args.stage == "seal": result=seal(config)
    else: result=score(config)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
