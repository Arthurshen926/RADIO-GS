"""Post-selection pose diagnostics; reads no labels and cannot alter source plans."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.utils.immutable_artifacts import sha256_file

def coverage(prior, sources, targets, diameter):
    poses=torch.stack([prior.cameras[f].world_to_camera for f in sources]).double()
    centers=-torch.einsum('nij,nj->ni',poses[:,:3,:3].transpose(1,2),poses[:,:3,3])
    rows=[]
    for f in targets:
        t=prior.cameras[f].world_to_camera.double();c=-t[:3,:3].T@t[:3,3]
        angle=torch.rad2deg(torch.acos((poses[:,2,:3]@t[2,:3]).clamp(-1,1)))
        distance=(centers-c).norm(dim=1)
        rows.append(dict(frame=f,nearest_direction_degrees=float(angle.min()),nearest_direction_source=sources[int(angle.argmin())],nearest_center_distance=float(distance.min()),nearest_center_over_original_source_diameter=float(distance.min()/diameter)))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    base=Path('output');root=base/'lerf_pose_covered_pipeline_20260917'/a.scene
    planpath=root/'source_plan.json';plan=json.loads(planpath.read_text())
    gp=base/'geometry_only_repair_final_20260914'/a.scene/'report.json';g=json.loads(gp.read_text());assert sha256_file(gp)==plan['geometry_report_sha256']
    prior=MogeGeometryPrior(g['source_report'],g['source_report_sha256'])
    evaluation=base/'lerf_ropefix_multiscale_pipeline_20260917'/a.scene/'text_evaluation.json'
    # Only the frozen evaluation camera IDs are used, after source selection.
    targets=json.loads(evaluation.read_text())['annotation_frames']
    before=plan['original_train_frames'];after=plan['train_frames'];assert not set(after)&set(targets)
    assert set(after)<=set(g['mapping_frames']) and set(after)<=set(plan['eligible_frames'])
    poses=torch.stack([prior.cameras[f].world_to_camera for f in before]).double();c=-torch.einsum('nij,nj->ni',poses[:,:3,:3].transpose(1,2),poses[:,:3,3]);diameter=torch.cdist(c,c).max();assert diameter>0
    report=dict(schema='source_pose_coverage.v1',scene=a.scene,geometry_mapping_views=len(g['mapping_frames']),registered_cameras=len(prior.cameras),old_train_views=len(before),new_train_views=len(after),source_plan_sha256=sha256_file(planpath),geometry_report_sha256=sha256_file(gp),evaluation_camera_list_sha256=sha256_file(evaluation),implementation_sha256=sha256_file(__file__),target_labels_opened=False,selection_uses_target_cameras=False,protocol='post-selection camera-only diagnostic; source plan precommitted; expanded input and training budget, not a matched architecture-only ablation',before=coverage(prior,before,targets,diameter),after=coverage(prior,after,targets,diameter))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,a.output.with_suffix('.source.py'));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
