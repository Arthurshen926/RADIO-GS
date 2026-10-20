"""Keep both hypothesis controls and every task; diagnose identity vs extent."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from radio_gs.query_field.bounded_experiment import tree_bytes
from radio_gs.query_field.shared_storage import load_field_v2
from radio_gs.query_field.unified_five_benchmark import read
from radio_gs.query_field.unified_hypothesis_study import store_for,seal,RULES
from radio_gs.query_field.storage import sha256


def bootstrap_difference(a,b):
    values=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(20261016)
    sample=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return dict(delta_pp=float(values.mean()*100),interval95_pp=(np.quantile(sample,[.025,.975])*100).tolist(),
        scenes=len(values),improved=int((values>1e-10).sum()),regressed=int((values< -1e-10).sum()),
        bootstrap_samples=2000,bootstrap_unit="scene (case for NVOS)",independent_test=False)


def read_arm(folder,plan):
    root=folder/"predictions";lerf=[];scannet=[];nvos=[]
    for job in plan["jobs"]["lerf"]:
        data=read(root/"lerf"/job["name"]/"evaluation/evaluation.json")["metrics"]
        lerf.append(dict(scene=job["name"],native_iou=data["native_generic_relevance"]["iou"],
            member_iou=data["shared_membership"]["iou"],member_ap=data["shared_membership"]["ap"],
            subset_iou=data["shared_membership"]["selected"]["miou"]))
    for job in plan["jobs"]["scannet"]:
        data=read(root/"scannet"/job["name"]/"evaluation.json")["metrics"]
        scannet.append(dict(scene=job["name"],native={s:d["miou"] for s,d in data["native"].items()},
                            member={s:d["miou"] for s,d in data["member"].items()}))
    for job in plan["jobs"]["nvos"]:
        data=read(root/"nvos"/job["name"]/"evaluation.json")["results"][0]
        nvos.append(dict(scene=job["name"],response_iou=data["retained_response"]["foreground_iou"],
                         ap=data["retained_response"]["average_precision"],subset_iou=data["foreground_iou"]))
    uqis=read(root/"uqis/evaluation.json")
    if (len(lerf),len(nvos),len(scannet),uqis["query_count"],uqis["target_count"])!=(4,8,10,268,67):raise ValueError("incomplete cohort")
    per_scene=dict(lerf=lerf,nvos=nvos,scannet=scannet)
    macro=dict(lerf={k:float(np.mean([r[k] for r in lerf])) for k in ["native_iou","member_iou","member_ap","subset_iou"]},
        scannet={a:{s:float(np.mean([r[a][s] for r in scannet])) for s in ["19","15","10"]} for a in ["native","member"]},
        nvos={k:float(np.mean([r[k] for r in nvos])) for k in ["response_iou","ap","subset_iou"]},
        uqis={m:dict(ap=d["scene_macro"]["average_precision"],subset_iou=uqis["explicit_selection"]["modalities"][m]["scene_macro"]["iou"],
                    empty_selections=uqis["explicit_selection"]["modalities"][m]["empty_selections"])
            for m,d in uqis["modalities"].items()})
    return dict(macro=macro,per_scene=per_scene,uqis=uqis)


def identity_audit(root,plan,arms):
    pkg=Path(plan["uqis_package"]);truth=read(pkg/"evaluator/targets.json")
    targets={qid:t for t in truth["targets"] for qid in t["queries"].values()}
    rows=[]
    candidate_sets={}
    for job in plan["jobs"]["uqis"]:
        scene=job["name"];domain=next(d for d in truth["scene_domains"] if d["scene_id"]==scene)
        xyz=np.load(pkg/"method"/domain["mesh_xyz_path"]);labels=np.load(pkg/"evaluator"/domain["mesh_instance_ids_path"])
        field,_=load_field_v2(job["field"],store_root=root/"blobs")
        _,index=cKDTree(xyz).query(field.xyz.numpy(),workers=4);gaussian_labels=labels[index]
        for rule in RULES:
            folder=root/("hypotheses_"+rule)/"predictions/uqis"
            member={q["query_id"]:q for q in arms[rule]["uqis"]["explicit_selection"]["per_query"]}
            for query in read(folder/"receipts"/(scene+".json"))["queries"]:
                payload=torch.load(folder/"gaussians"/(query["query_id"]+".pt"),map_location="cpu",weights_only=True)
                ids=payload["candidates"].numpy();weights=payload["candidate_weights"].numpy()
                if rule=="peak":candidate_sets[query["query_id"]]=ids.copy()
                elif not np.array_equal(ids,candidate_sets[query["query_id"]]):
                    raise ValueError("matched controls generated different UQIS candidates")
                hits=gaussian_labels[ids]==targets[query["query_id"]]["instance_id"]
                rows.append(dict(rule=rule,scene=scene,modality=query["modality"],query_id=query["query_id"],
                    any_target=bool(hits.any()),chosen_target=bool(hits[weights.argmax()]) if len(weights) else False,
                    candidates=len(ids),dominant_target=member[query["query_id"]]["target_is_unique_dominant_instance"],
                    iou=member[query["query_id"]]["iou"]))
        del field
    summary={rule:{modality:dict(queries=len(rs),candidate_recall_count=sum(q["any_target"] for q in rs),
        chosen_target_count=sum(q["chosen_target"] for q in rs),dominant_target_count=sum(q["dominant_target"] for q in rs),
        correct_anchor_wrong_extent=sum(q["chosen_target"] and not q["dominant_target"] for q in rs))
        for modality in ["text","image","point_2d","point_3d"]
        for rs in [[q for q in rows if q["rule"]==rule and q["modality"]==modality]]} for rule in RULES}
    return dict(summary=summary,queries=rows,gt_diagnostic_only=True,no_prediction_changes=True)


def summarize(config):
    root=Path(config["study_root"]);plan=read(root/"cohort/plan.json")
    for rule in RULES:
        if not (root/("hypotheses_"+rule)/"scoring_complete.json").exists():raise ValueError("both full-five controls must finish")
    seal(config)
    for family in ("lerf","scannet"):
        for job in plan["jobs"][family]:
            pair=[read(root/("hypotheses_"+rule)/"predictions"/family/job["name"]/"prediction.json") for rule in RULES]
            if family=="scannet":
                if [q["candidates"] for q in pair[0]["queries"]]!=[q["candidates"] for q in pair[1]["queries"]]:
                    raise ValueError("matched category candidates differ")
            else:
                for a,b in zip(pair[0]["diagnostics"],pair[1]["diagnostics"]):
                    if any(a[k]!=b[k] for k in ("text","candidate_count","peak_scores","pooled_scores","hypothesis_sizes")):
                        raise ValueError("matched text hypothesis statistics differ")
    arms={"baseline":read_arm(root/"cohort",plan),**{rule:read_arm(root/("hypotheses_"+rule),plan) for rule in RULES}}
    contrasts={}
    for left,right in [("peak","baseline"),("pooled","baseline"),("pooled","peak")]:
        rows=[]
        for family,metrics in [("lerf",["member_iou","member_ap","subset_iou"]),("nvos",["response_iou","ap","subset_iou"])]:
            for metric in metrics:
                rows.append(dict(task=family,metric=metric,**bootstrap_difference(
                    [r[metric] for r in arms[left]["per_scene"][family]],[r[metric] for r in arms[right]["per_scene"][family]])))
        for metric in ["19","15","10"]:
            rows.append(dict(task="scannet",metric="native_miou_"+metric,**bootstrap_difference(
                [r["native"][metric] for r in arms[left]["per_scene"]["scannet"]],
                [r["native"][metric] for r in arms[right]["per_scene"]["scannet"]])))
        for modality in ["text","image","point_2d","point_3d"]:
            for metric in ["ap","subset_iou"]:
                values=[]
                for arm in [left,right]:
                    uq=arms[arm]["uqis"]
                    per_scene=(uq["modalities"][modality]["per_scene"] if metric=="ap" else uq["explicit_selection"]["modalities"][modality]["per_scene"])
                    key="average_precision" if metric=="ap" else "iou"
                    values.append([per_scene[j["name"]][key] for j in plan["jobs"]["uqis"]])
                rows.append(dict(task="uqis",modality=modality,metric=metric,**bootstrap_difference(*values)))
        contrasts[left+"_minus_"+right]=rows
    meta=store_for(config,"hypothesis_comparison")
    identity=identity_audit(root,plan,arms);meta.json("identity_audit.json",identity)
    result=dict(schema="radio_gs.unified_hypothesis_results.v1",five_benchmarks_complete=True,
        rules=list(RULES),macro={k:v["macro"] for k,v in arms.items()},paired_differences=contrasts,
        identity=identity["summary"],source_hash=sha256(__file__),design=read(meta.root/"design.json"),
        all_prediction_barrier_sha256=sha256(meta.root/"prediction_barrier.json"),new_model_bytes=0,
        matched_candidates_and_component_statistics_verified=True,
        study_bytes=tree_bytes(root),accuracy_promotion=False,independent_final_test=False,
        per_scene={k:v["per_scene"] for k,v in arms.items()})
    meta.json("summary.json",result)
    with Path("docs/experiments/2026-10-15-unified-hypothesis-results.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/query_field/unified_five_benchmark.json")
    args=p.parse_args();result=summarize(read(args.config));print(json.dumps(dict(macro=result["macro"],identity=result["identity"]),indent=2))
