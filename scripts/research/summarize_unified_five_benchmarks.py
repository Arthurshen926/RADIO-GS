"""Predeclare typed readouts, then report the entire common compact cohort."""
import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.query_field.bounded_experiment import tree_bytes
from radio_gs.query_field.storage import sha256
from radio_gs.query_field.unified_five_benchmark import control, read, seal, TASKS


CONTRACT = dict(
    tasks=list(TASKS), all_teachers=["SigLIP2","DINOv2","SAM"],
    primary_readouts=dict(lerf2d="shared_membership.iou",lerf3d_local_projection="shared_membership.selected.miou",
        nvos="retained_response.foreground_iou",scannet_ovs="native complete-category partition",
        uqis="all four modalities: AP and actual selected-set IoU"),
    category_readout="Fixed UnifiedCompactField.category_partition is native cosine argmax, including void support. Raw independent binary member competition is a diagnostic, not a partition.",
    diagnostic_controls=["native LERF relevance", "raw independent ScanNet member argmax", "NVOS actual subset projection"],
    promotion_rule="No promotion without all five complete; show every per-task delta, regressions and unsupported/empty queries. No weighted scalar winner or benchmark-specific router.",
    parameter_selection="source-development only; no task GT selects scenes/checkpoints/thresholds",
    independent_final_test=False, selected_projection_is_true_3d_ground_truth=False,
    controls_are_not_multiple_systems=True,
    limitations=["scene-specific learned weights, common recipe; not cross-scene shared weights",
        "all teachers actively supervise one latent; does not prove complete functional transfer",
        "v10 has different source budgets and teacher/input contracts; comparisons are system gaps, not isolated architecture effects"])


def declare(config):
    store=control(config)
    if (store.root / "prediction_barrier.json").exists() or (store.root / "scoring_complete.json").exists():
        raise ValueError("declare readouts before complete prediction seal/scoring")
    path=store.root / "feedback_contract.json"
    if path.exists():
        if read(path)!=CONTRACT:raise ValueError("readout contract changed")
    else:store.json("feedback_contract.json",CONTRACT)
    return CONTRACT


def summarize(config):
    store=control(config);plan=read(store.root / "plan.json")
    if read(store.root / "feedback_contract.json")!=CONTRACT:raise ValueError("readouts differ")
    if not (store.root / "scoring_complete.json").exists():raise ValueError("all-five scoring is incomplete")
    seal(config)
    root=store.root / "predictions"
    lerf=[];nvos=[];scannet=[]
    for job in plan["jobs"]["lerf"]:
        d=read(root / "lerf" / job["name"] / "evaluation/evaluation.json")["metrics"]
        lerf.append(dict(scene=job["name"],native_iou=d["native_generic_relevance"]["iou"],
            native_ap=d["native_generic_relevance"]["ap"],member_iou=d["shared_membership"]["iou"],
            member_ap=d["shared_membership"]["ap"],subset_iou=d["shared_membership"]["selected"]["miou"],
            query_views=d["shared_membership"]["observations"],missing=d["shared_membership"]["selected"]["missing"]))
    for job in plan["jobs"]["nvos"]:
        d=read(root / "nvos" / job["name"] / "evaluation.json")["results"][0]
        nvos.append(dict(scene=job["name"],response_iou=d["retained_response"]["foreground_iou"],
            ap=d["retained_response"]["average_precision"],subset_iou=d["foreground_iou"],selected=d["selected"]))
    for job in plan["jobs"]["scannet"]:
        d=read(root / "scannet" / job["name"] / "evaluation.json")
        scannet.append(dict(scene=job["name"],observed_vertex_fraction=d["observed_vertex_fraction"],
            **{arm:{k:v["miou"] for k,v in d["metrics"][arm].items()} for arm in ["native","member"]}))
    uqis=read(root / "uqis/evaluation.json")
    if (len(lerf),len(nvos),len(scannet),uqis["query_count"],uqis["target_count"])!=(4,8,10,268,67):
        raise ValueError("complete denominators differ")
    macro=dict(lerf={k:float(np.mean([r[k] for r in lerf])) for k in ["native_iou","native_ap","member_iou","member_ap","subset_iou"]},
        nvos={k:float(np.mean([r[k] for r in nvos])) for k in ["response_iou","ap","subset_iou"]},
        scannet={a:{k:float(np.mean([r[a][k] for r in scannet])) for k in ["19","15","10"]} for a in ["native","member"]},
        uqis={m:dict(ap=uqis["modalities"][m]["scene_macro"]["average_precision"],
            subset_iou=uqis["explicit_selection"]["modalities"][m]["scene_macro"]["iou"],
            empty_selections=uqis["explicit_selection"]["modalities"][m]["empty_selections"])
            for m in ["text","image","point_2d","point_3d"]})
    baseline_path=Path("output/query_field_20261007/frozen_v10/baseline_a.json")
    baseline=read(baseline_path)["macro"];rows=[]
    def add(task,metric,candidate,v10):
        rows.append(dict(task=task,metric=metric,candidate=candidate,v10_readback=v10,delta_pp=100*(candidate-v10)))
    add("lerf2d","member response IoU",macro["lerf"]["member_iou"],baseline["lerf"]["semantic_iou"])
    add("lerf2d","member response AP",macro["lerf"]["member_ap"],baseline["lerf"]["ap"])
    add("lerf3d_local_projection","actual subset projection IoU",macro["lerf"]["subset_iou"],baseline["lerf"]["selected_projection_iou"])
    for k in ["19","15","10"]:add("scannet_ovs","category mIoU "+k,macro["scannet"]["native"][k],baseline["scannet"]["miou_"+k])
    add("nvos","signed response IoU",macro["nvos"]["response_iou"],baseline["nvos"]["iou"])
    add("nvos","signed response AP",macro["nvos"]["ap"],baseline["nvos"]["ap"])
    for m in macro["uqis"]:
        add("uqis/"+m,"AP",macro["uqis"][m]["ap"],baseline["uqis"][m+"_ap"])
        add("uqis/"+m,"actual subset IoU",macro["uqis"][m]["subset_iou"],baseline["uqis"][m+"_selected_iou"])
    unsupported={m:0 for m in macro["uqis"]}
    for job in plan["jobs"]["uqis"]:
        for q in read(root / "uqis/receipts" / (job["name"]+".json"))["queries"]:
            unsupported[q["modality"]]+=q["status"]!="ok"
    fields=[read(Path(config["study_root"]) / f["run"] / "training.json") for f in plan["fields"]]
    report=dict(schema="radio_gs.unified_five_results.v1",five_benchmarks_complete=True,
        architecture=config["architecture"],all_teachers=True,all_fields=30,contract=CONTRACT,
        comparison=rows,macro=macro,per_scene=dict(lerf=lerf,nvos=nvos,scannet=scannet),
        uqis_unsupported=unsupported,uqis_identity_diagnosis=uqis["explicit_selection"]["identity_diagnosis"],
        uqis_paired_actual_sets=uqis["explicit_selection"]["paired_query_robustness"],
        teacher_transfer_diagnostics=[dict(scene=f["scene"],selected_step=f["selected_step"],
            selected_validation=f["selected_validation"],teacher_gradient_audit=f["teacher_gradient_audit"]) for f in fields],
        model_payload_bytes=sum(f["complete_field_payload_bytes"] for f in fields),
        study_file_bytes=tree_bytes(store.study),query_encoder_cost_excluded=True,accuracy_promotion=False,
        baseline_sha256=sha256(baseline_path),plan_sha256=sha256(store.root / "plan.json"),
        feedback_contract_sha256=sha256(store.root / "feedback_contract.json"))
    store.json("summary.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=["declare","summarize"])
    p.add_argument("--config",default="configs/query_field/unified_five_benchmark.json");a=p.parse_args()
    result=(declare if a.stage=="declare" else summarize)(read(a.config))
    print(json.dumps(result,indent=2))
