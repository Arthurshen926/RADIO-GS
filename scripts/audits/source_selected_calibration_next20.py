"""Source-only actual-selection calibration; fixed bounded pilot, no target inputs.

Reserve last four source frames for readout validation. They were used to build
features: this is not independent scene/view validation of the field.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_threshold import fit_threshold_ridge
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def grouped_predictions(q, targets):
    parents=list(range(len(q)))
    def find(i):
        while parents[i]!=i:
            i=parents[i]
        return i
    sim=q@q.T
    for i in range(len(q)):
        for j in range(i):
            if sim[i,j]>=.95:
                parents[find(i)]=find(j)
    groups=[find(i) for i in range(len(q))]
    pred=torch.empty(len(q))
    for i in range(len(q)):
        keep=torch.tensor([g!=groups[i] for g in groups])
        if keep.sum()<2:
            raise ValueError("too few independent source concepts")
        w,b=fit_threshold_ridge(q[keep],targets[keep],1.)
        pred[i]=(q[i]@w+b).clamp(targets[keep].min(),targets[keep].max())
    return pred,groups


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--scene",required=True);a=p.parse_args()
    torch.set_num_threads(4)
    root=Path("output/full_field_rebuild_next19")/a.scene/"reconstruction/field"
    out=Path("output/source_selected_calibration_next20")/a.scene
    out.mkdir(parents=True,exist_ok=False)
    registration=dict(scene=a.scene,offsets=[-.08,-.04,-.02,0.,.02,.04,.08],additional_candidate="previous exact probability target",objective="maximize minimum(probability IoU, selected IoU) on source training views",reserve="last four source frames; field already trained on them",group_cosine=.95,ridge=1.,target_inputs_used=False,source_masks_are_pseudo=True,promotion="require both probability and selected mean nonregression on reserved source frames under grouped unseen-query prediction",element_threshold=.5,selected_byte_threshold=10)
    (out/"registration.json").write_text(json.dumps(registration,indent=2))
    fp=root/"bounded_fallback_admission/field.pt";receipt=json.loads((fp.parent/"report.json").read_text())
    assert sha256_file(fp)==receipt["field_sha256"]
    payload=load(fp);model=ElementRetrievalField(**payload["architecture"]).cuda();model.load_state_dict(payload["state_dict"]);model.requires_grad_(False)
    meta=json.loads((root/"exact_render_query_threshold96/report.json").read_text())
    assert sha256_file(root/"episodes.pt")==meta["episode_sha256"]
    assert sha256_file(root/"source_grounded_text.pt")==meta["raw_source_grounding_sha256"]
    data=load(root/"episodes.pt");raw=load(root/"source_grounded_text.pt")
    info=json.loads((root/"teachers.json").read_text());assert raw["frames"]==info["train_frames"] and not raw["benchmark_labels_used"]
    assert raw["geometry_binding"]==payload["geometry_binding"]
    qids=meta["query_indices"];q=F.normalize(data["queries"]["text"][qids].float(),dim=1)
    cos=model.native_cosine(q.cuda());obs=model.observed;scale=model.readout_log_scale["text"].clamp(-4,6).exp()
    current=model.native_query_thresholds["text"](q.cuda());old=torch.tensor(meta["history"][-1]["thresholds"],device="cuda")
    grid=torch.cat([(current[:,None]+torch.tensor(registration["offsets"],device="cuda")[None]).clamp(-1,1),old[:,None]],1)
    gp=Path("output/depth_opacity_repair_20260918")/a.scene/"geometry_report.json";gr=json.loads(gp.read_text());g=gr["results"]["independent_scale"]
    geo=FrozenGaussianGeometry.load(g["geometry"],expected_sha256=g["geometry_sha256"],device="cuda:0");geo.assert_binding(payload["geometry_binding"]);prior=MogeGeometryPrior(gr["source_report"],gr["source_report_sha256"])
    h,w=raw["probability"].shape[-2:];held=set(raw["frames"][-4:]);rows=[]
    def evaluate(qi,cuts,frames):
        scores=torch.sigmoid(scale*(cos[:,qi,None]-cuts[None]))*obs[:,None]
        colors=torch.cat([scores,obs[:,None].float()],1)
        result=[]
        for vi,frame in enumerate(raw["frames"]):
            if frame not in frames:continue
            target=raw["probability"][vi,qids[qi]].cuda()>.5
            if raw["scores"][vi,qids[qi]]<.7 or target.sum()<20:continue
            cam=prior.cameras[frame];k=cam.intrinsic.cuda().clone();k[0]*=w/cam.width;k[1]*=h/cam.height
            kw=dict(viewmats=cam.world_to_camera.cuda()[None],Ks=k[None],width=w,height=h,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            im,alpha,_=rasterization(means=geo.xyz,quats=geo.quaternions,scales=geo.scales,opacities=geo.opacities.flatten(),colors=colors,**kw)
            mass=im[0,...,-1];valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            prob=torch.where(valid[...,None],im[0,...,:-1]/mass[...,None].clamp_min(1e-8),0)
            metrics=[]
            for j in range(len(cuts)):
                selected=(scores[:,j]>.5)&obs
                if selected.any():
                    _,aa,_=rasterization(means=geo.xyz[selected],quats=geo.quaternions[selected],scales=geo.scales[selected],opacities=geo.opacities.flatten()[selected],colors=torch.ones(int(selected.sum()),1,device="cuda"),**kw)
                    mask=(aa[0,...,0]*255+.5).floor()>10
                else:mask=torch.zeros_like(target)
                def iou(pred):return float((pred&target).sum()/(pred|target).sum().clamp_min(1))
                metrics.append(dict(probability_iou=iou(prob[...,j]>.5),selected_iou=iou(mask),selected_elements=int(selected.sum())))
            result.append(dict(query_index=qids[qi],query=raw["queries"][qids[qi]],frame=frame,heldout=frame in held,metrics=metrics))
        return result
    targets=current.cpu().clone();eligible=[]
    for qi in range(len(q)):
        rr=evaluate(qi,grid[qi],set(raw["frames"]));rows.extend(rr)
        train=[r for r in rr if not r["heldout"]]
        if train:
            values=np.array([[min(m["probability_iou"],m["selected_iou"]) for m in r["metrics"]] for r in train]).mean(0)
            # Prefer unchanged current boundary on ties.
            best=3 if values[3]>=values.max()-1e-12 else int(values.argmax())
            targets[qi]=grid[qi,best].cpu();eligible.append(qi)
        print(a.scene,"grid",qi+1,"/",len(q),flush=True)
    # Do not fit pseudo-targets for concepts with no training observations.
    ids=torch.tensor(eligible);pred,groups=grouped_predictions(q[ids],targets[ids]);validation=[]
    for k,qi in enumerate(eligible):
        rr=evaluate(qi,torch.stack([current[qi],pred[k].cuda()]),held);validation.extend(rr)
    macro={name:{metric:float(np.mean([r["metrics"][i][metric] for r in validation])) for metric in ["probability_iou","selected_iou"]} for i,name in enumerate(["current","grouped_selection_fit"])}
    assert validation,"no reserved source observations"
    accepted=all(macro["grouped_selection_fit"][m]>=macro["current"][m] for m in macro["current"]) and any(macro["grouped_selection_fit"][m]>macro["current"][m]+1e-6 for m in macro["current"])
    report=dict(complete=True,registration=registration,macro=macro,source_accepted=accepted,independent_test=False,field_unchanged=True,query_ids=qids,eligible_query_positions=eligible,groups=groups,targets=targets.tolist(),grouped_predictions=pred.tolist(),grid=grid.cpu().tolist(),source_grid_observations=rows,reserved_observations=validation,field_sha256=receipt["field_sha256"],source_sha256=meta["raw_source_grounding_sha256"],implementation_sha256=sha256_file(__file__))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(a.scene,macro,"accepted",accepted,flush=True)


if __name__=="__main__":main()
