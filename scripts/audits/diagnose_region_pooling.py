"""Diagnostic only: expose nonlinear source region responses before compaction.

Region bank is a privileged training artifact, not an exported deployment map.
GT is used only for scoring and explicitly labelled best-threshold diagnostics.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from gsplat import rasterization
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_common import load_labels, ground_truth_masks


def threshold_diagnostic(score, truth):
    values, indices = score.flatten().sort(descending=True)
    y = truth.flatten()[indices].float()
    tp = y.cumsum(0)
    count = torch.arange(1, len(y)+1, device=y.device)
    ends = torch.cat([values[:-1] != values[1:], torch.ones(1, device=y.device, dtype=torch.bool)])
    iou = tp / (truth.sum() + count - tp).clamp_min(1)
    return dict(oracle_threshold_iou=float(iou[ends].max()), peak_hit=bool(y[0]),
                top_one_percent_precision=float(y[:max(1,len(y)//100)].mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    output = Path(f'paper/artifacts/{a.scene}_region_pooling_diagnostic_20260916.json')
    if output.exists(): raise FileExistsError(output)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    root = base / 'preserved_source_inventory_20260916' / a.scene
    info = json.loads((root/'teachers.json').read_text())
    receipt = json.loads((root/'episodes.json').read_text())
    assert sha256_file(root/'episodes.pt') == receipt['sha256']
    data = torch.load(root/'episodes.pt', weights_only=False)
    assert sha256_file(root/'teachers.json') == data['teacher_manifest_sha256']
    rec = info['teacher_files']['siglip']
    assert sha256_file(root/rec['path']) == rec['sha256']
    teacher = torch.load(root/rec['path'], weights_only=False)['descriptors'].float().to(a.device)
    gr = json.loads((base/'geometry_only_repair_final_20260914'/a.scene/'report.json').read_text())
    g = gr['results']['independent_scale']
    geometry = FrozenGaussianGeometry.load(g['geometry'], expected_sha256=g['geometry_sha256'], device=a.device)
    geometry.assert_binding(data['geometry_binding'])
    prior = MogeGeometryPrior(gr['source_report'], gr['source_report_sha256'])
    identity = json.loads((root/'native_query_encoder.json').read_text())
    modelpath = root/'scale_affinity_v1_direct_control'/'field.pt'
    field, _ = load_element_retrieval_field(modelpath, expected_sha256=sha256_file(modelpath), geometry=geometry, expected_query_encoder=identity, device=a.device)
    trainviews = {info['frames'].index(f) for f in info['train_frames']}
    ids = torch.tensor([int(v) in trainviews for v in data['proposal_views']])
    weights = data['region_probability'][ids].float().to(a.device)
    basis = field.native_decoder.weight
    descriptors = F.normalize((F.normalize(teacher[ids],dim=-1)@basis)@basis.T,dim=-1)
    labelroot = Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label')
    annotations, categories, sh, sw = load_labels(labelroot,a.scene)
    assert not set(annotations)&set(info['train_frames'])
    queries = NativeThreeQueryEncoder(identity,a.device).encode_text(categories).half().float()
    with torch.no_grad():
        region_scores = (descriptors@F.normalize(queries,dim=-1).T+1)/2
        responses = {'field_native':(field.native_cosine(queries)+1)/2,
                     'region_response_mean':weights.T@region_scores/weights.sum(0)[:,None].clamp_min(1e-8)}
        maximum = []
        for start in range(0,len(field.latent),1024):
            maximum.append((weights[:,start:start+1024,None]*region_scores[:,None,:]).amax(0))
        responses['region_response_max'] = torch.cat(maximum)
        colors = torch.cat([v*field.observed[:,None] for v in responses.values()]+[field.observed.float()[:,None]],1)
        rows = {k:[] for k in responses}
        height = round(sh*480/sw)
        for frame, objects in sorted(annotations.items()):
            camera = prior.cameras[frame]; k = camera.intrinsic.to(a.device).clone()
            k[0]*=480/camera.width; k[1]*=height/camera.height
            image,alpha,_ = rasterization(means=geometry.xyz,quats=geometry.quaternions,scales=geometry.scales,
                opacities=geometry.opacities[:,0],colors=colors,viewmats=camera.world_to_camera.to(a.device)[None],
                Ks=k[None],width=480,height=height,near_plane=.001*prior.scale,far_plane=100*prior.scale,packed=False)
            mass=image[0,...,-1]; valid=(mass/alpha[0,...,0].clamp_min(1e-8)>.5)&(alpha[0,...,0]>1e-8)
            maps=image[0,...,:-1]/mass[...,None].clamp_min(1e-8)
            truth=ground_truth_masks(objects,categories,height,480,sh,sw)
            for ci,c in enumerate(categories):
                target=truth[c].to(a.device)
                if not target.any(): continue
                for j,key in enumerate(responses):
                    score=torch.where(valid,maps[...,j*len(categories)+ci],torch.zeros_like(mass))
                    rows[key].append(dict(frame=frame,category=c,**binary_ranking_metrics(score,target),
                                          **threshold_diagnostic(score,target)))
        metrics=['average_precision','oracle_threshold_iou','peak_hit','top_one_percent_precision']
        macro={k:{m:sum(r[m] for r in v)/len(v) for m in metrics} for k,v in rows.items()}
    result=dict(schema='region_pooling_diagnostic.v1',diagnostic_only=True,not_deployment=True,
                source_episode_sha256=receipt['sha256'],field_sha256=sha256_file(modelpath),
                geometry_sha256=g['geometry_sha256'],implementation_sha256=sha256_file(__file__),
                labels={str(p):sha256_file(p) for p in sorted((labelroot/a.scene).glob('*.json'))},
                contract='same support and native SVD64 basis; rendered affine cosine; no calibrated IoU claim; target RGB never encoded',
                oracle_threshold_uses_gt=True,macro=macro,observations=rows)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(macro),flush=True)


if __name__ == '__main__': main()
