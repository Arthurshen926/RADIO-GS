"""Reference-only diagnostic of why disconnected component evidence abstains."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file
from scripts.audits.prompt_transfer_20260919_evaluate import refine
from scripts.audits.nvos_component_evidence_runtime_next31_v2 import complete_reference_query
import radio_gs.querying.component_reference_evidence_numeric as numeric


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    mp=Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json')
    manifest=json.loads(mp.read_text());scene=next(s for s in manifest['scenes'] if s['scene_id']==a.scene)
    forbidden={str(Path(scene['prompt']['negative_path']).resolve())}
    for f in scene['frames']:
        if f['frame_id'] in scene['evaluation_frame_ids']:
            forbidden.update(str(Path(f[k]).resolve()) for k in ('rgb_path','annotation_rgb_path','gt_mask_path') if f.get(k))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=str(Path(os.fsdecode(args[0])).resolve())
            if name in forbidden:raise RuntimeError('Reference-only diagnostic forbids target RGB/GT and negative scribble')
            if any(x in name for x in ['teacher_sam.pt','teacher_siglip.pt','teacher_radio.pt','teacher_dino.pt','episodes.pt']):
                raise RuntimeError('Cold diagnostic cannot read training teachers')
    sys.addaudithook(guard)
    parent=Path('output/nvos_ropefix_pipeline_20260917')/('horns_center' if a.scene=='horns_left' else a.scene)
    gp=parent/'frozen_geometry/report.json';gr=json.loads(gp.read_text());g=gr['results']['independent_scale']
    geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=a.device)
    prior=MogeGeometryPrior(gr['source_report'],gr['source_report_sha256']);identity=json.loads((parent/'field/native_query_encoder.json').read_text())
    qrpath=Path('output/nvos_positive_queries_20260918')/a.scene/'receipt.json';qr=json.loads(qrpath.read_text())
    assert not qr['negative_scribble_opened'];crop=qr['image_query_crop'];assert sha256_file(crop['path'])==crop['sha256']
    encoder=NativeThreeQueryEncoder(identity,a.device);q=encoder.encode_image(Image.open(crop['path']).convert('RGB'))
    encoder.vision=None;gc.collect();torch.cuda.empty_cache()
    fp=parent/'field/image_supervised_native_residual_measured_support512/field.pt';fr=json.loads(fp.with_name('report.json').read_text())
    field,_=load_element_retrieval_field(fp,expected_sha256=fr['field_sha256'],geometry=geo,expected_query_encoder=identity,device=a.device)
    positive=np.asarray(Image.open(qr['official_scribbles']['positive']['path']))
    positive=positive>127 if positive.ndim==2 else np.any(positive>127,axis=2)
    sam=np.asarray(Image.open(qr['mask']))>127;assert sha256_file(qr['mask'])==qr['mask_sha256']
    width,height=qr['prompt_raster_wh'];reference=next(f for f in scene['frames'] if f['frame_id']==scene['prompt']['frame_id'])
    camera=prior.cameras[int(reference['canonical_index'])];positive=torch.from_numpy(positive);sam=torch.from_numpy(sam)
    evidence={};original=numeric.retain_numerically_checked_components
    def capture(labels,candidate,retained,foreground_mass,visibility_mass,**kw):
        output,record=original(labels,candidate,retained,foreground_mass,visibility_mass,**kw)
        n=int(labels.max())+1;ids=labels[candidate].unique()
        def sum_component(value):return torch.bincount(labels[candidate],weights=value[candidate].double(),minlength=n)
        fg=sum_component(foreground_mass);vis=sum_component(visibility_mass);size=sum_component(torch.ones_like(foreground_mass))
        old=sum_component(retained.float())>0;purity=fg/vis.clamp_min(1e-12);removed=ids[~old[ids]]
        order=removed[fg[removed].argsort(descending=True)]
        strong=(foreground_mass/visibility_mass.clamp_min(1e-12)>=.9)&(foreground_mass>=1)&candidate&~retained
        evidence.update(candidate_components=len(ids),retained_components=int(old[ids].sum()),
            removed_candidate_components=len(removed),
            removed_components_no_effective_reference_visibility=int((vis[removed]<1).sum()),
            removed_visible_components_core_purity_below_half=int(((vis[removed]>=1)&(purity[removed]<.5)).sum()),
            removed_visible_components_core_purity_half_to_point9=int(((vis[removed]>=1)&(purity[removed]>=.5)&(purity[removed]<.9)).sum()),
            removed_visible_components_core_purity_point9_or_more=int(((vis[removed]>=1)&(purity[removed]>=.9)).sum()),
            removed_fraction_of_candidate_reference_core_mass=float(fg[removed].sum()/fg[ids].sum().clamp_min(1e-12)),
            pure_individual_elements_in_removed_components=int(strong.sum()),
            pure_individual_elements_core_mass=float(foreground_mass[strong].sum()),
            largest_removed_core_evidence=[dict(component=int(i),elements=int(size[i]),core_mass=float(fg[i]),visibility_mass=float(vis[i]),core_purity=float(purity[i])) for i in order[:20]],
            correction=record)
        return output,record
    numeric.retain_numerically_checked_components=capture
    try:
        with torch.no_grad():
            base=field.retrieve(q,'image')['scores']
            base,_=refine(base,field.observed_for('image'),geo,camera,prior,width,height,positive,sam,'prompt_sam')
            _,r=complete_reference_query(base,field.observed_for('image'),geo,camera,prior,width,height,positive,sam,'reference_core_component',field,base_stage='prompt_sam',enable_component_evidence=True)
    finally:numeric.retain_numerically_checked_components=original
    result=dict(scene=a.scene,reference_only=True,target_labels_or_rgb_read=False,diagnostic_only=True,
        method_parameters_changed=False,field_sha256=fr['field_sha256'],geometry_report_sha256=sha256_file(gp),
        query_receipt_sha256=sha256_file(qrpath),implementation_sha256=sha256_file(__file__),
        evidence=evidence,reference_after_filter=r['reference_after_filter'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
