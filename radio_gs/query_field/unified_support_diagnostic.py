"""Whole-cohort support/conditioning audit on held-out source views, no task GT."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bounded_experiment import BoundedExperiment
from .contracts import Support
from .functional_training import FunctionalSource
from .shared_storage import load_field_v2
from .storage import sha256
from .unified_five_benchmark import read
from .unified_training import sampled_support


@torch.no_grad()
def audit_field(field, source, count=24):
    device=field.xyz.device
    generator=torch.Generator().manual_seed(20261016)
    indices=np.linspace(0,len(source.examples)-1,min(count,len(source.examples)),dtype=int)
    episodes=[source.sample(int(i),generator,32,device) for i in indices]
    queries=torch.stack([e[2]["semantic"] for e in episodes])
    native=field.semantic.score(queries).masked_fill(~field.observed[:,None],-torch.inf)
    ids=native.topk(field.candidate_count,dim=0).indices
    results={k:[] for k in ("ray_explicit","ray_inferred_flag","retrieved_explicit_flag","retrieved_inferred")}
    for column,(rows,op,_,_) in enumerate(episodes):
        ray=sampled_support(op,rows)
        candidates=ids[:,column]
        retrieved=Support(candidates,(10*native[candidates,column]).softmax(0))
        target=torch.cat([torch.ones(32,device=device),torch.zeros(32,device=device)])
        for key,support,explicit in [("ray_explicit",ray,True),("ray_inferred_flag",ray,False),
                ("retrieved_explicit_flag",retrieved,True),("retrieved_inferred",retrieved,False)]:
            probability=field.member_logits(support,field.default_scale,rows.to(device),explicit=explicit).sigmoid()
            predicted=torch.sparse.mm(op,probability[:,None]).flatten()
            own=field.member_logits(support,field.default_scale,support.indices,explicit=explicit).sigmoid()
            results[key].append(dict(bce=float(F.binary_cross_entropy(predicted.clamp(1e-6,1-1e-6),target)),
                positive_mean=float(predicted[:32].mean()),negative_mean=float(predicted[32:].mean()),
                true_positive=int((predicted[:32]>.5).sum()),false_positive=int((predicted[32:]>.5).sum()),
                positive_count=32,support_self_mean=float((own*support.weights/support.weights.sum()).sum()),
                support_self_below_half=float((own<.5).float().mean())))
    return {key:dict(episodes=len(rows),mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0]},
        sampled_iou=sum(r["true_positive"] for r in rows)/sum(r["positive_count"]+r["false_positive"] for r in rows))
        for key,rows in results.items()}


def run(config):
    plan=read(Path(config["study_root"])/"cohort/plan.json")
    store=BoundedExperiment(config["study_root"],"support_diagnostic",config["storage"])
    store.json("design.json",dict(source_sha256=sha256(__file__),plan_sha256=sha256(Path(config["study_root"])/"cohort/plan.json"),
        episodes=24,seed=20261016,pairs=32,benchmark_queries_used=False,benchmark_labels_used=False,
        no_parameter_update=True,conditions=["same ray support with flag true/false","same retrieved support with flag true/false"],
        original_weights=True,all_thirty_scenes=True))
    reports=[]
    for entry in plan["fields"]:
        field,manifest=load_field_v2(entry["field"],"cuda",store_root=Path(config["study_root"])/"blobs")
        source=FunctionalSource(entry["observations"],field.observed,"development")
        result=dict(run=entry["run"],family=entry["family"],state_identity=manifest["state_identity"],
                    source_frames=[v["frame"] for v in source.views],metrics=audit_field(field,source))
        store.json(entry["run"]+".json",result);reports.append(result)
        print(entry["run"],json.dumps({k:(v["sampled_iou"],v["mean"]["support_self_mean"]) for k,v in result["metrics"].items()}),flush=True)
        del field,source;torch.cuda.empty_cache()
    summary={family:{condition:{metric:float(np.mean([r["metrics"][condition]["mean"][metric] for r in reports if r["family"]==family]))
            for metric in reports[0]["metrics"][condition]["mean"]} for condition in reports[0]["metrics"]}
            for family in ("lerf","nvos","scannet","uqis")}
    store.json("summary.json",dict(families=summary,fields=len(reports),benchmark_accuracy=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/query_field/unified_five_benchmark.json")
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.4);run(read(a.config))
