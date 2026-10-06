"""Build exact authority for the declared provided-mesh track, without a VFM."""
import argparse,json,inspect
from pathlib import Path
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.geometry.registered_source_prior import RegisteredSourcePrior as MogeGeometryPrior
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer
from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
from radio_gs.rendering.sparse_marginal_authority import SparseExactMarginalAuthorityWriter
import radio_gs.rendering.sparse_marginal_authority as authority_module
from radio_gs.utils.immutable_artifacts import sha256_file

def build(args):
    path=Path(args.geometry_report)
    if sha256_file(path)!=args.expected_report_sha256:
        raise ValueError("geometry report identity differs")
    report=json.loads(path.read_text())
    if report.get('input_track')!='provided_unlabelled_mesh':raise ValueError('Explicit provided-mesh input track required')
    frames=[int(x) for x in args.frame_ids.split(',')]
    if len(set(frames))!=len(frames) or not set(frames)<=set(report['mapping_frames']):
        raise ValueError("authority frames must be unique mapping-source frames")
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    prior=MogeGeometryPrior(report['source_report'],report['source_report_sha256'])
    g=report['results']['independent_scale']
    model=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device=args.device)
    binding=model.fingerprint()
    metadata={'geometry_source':'provided_unlabelled_registered_mesh_v1',
        'frozen_geometry_binding':binding,'frozen_geometry_sha256':g['geometry_sha256'],
        'frozen_geometry_report_sha256':args.expected_report_sha256,
        'geometry_trainable_during_semantics':False,'feature_confidence_contract':model.feature_confidence_contract,
        'feature_height':args.height,'feature_width':args.width,
        'benchmark_images_opened':False,'benchmark_masks_opened':False,'text_queries_opened':False,
        'feature_models_loaded':[],'semantic_validity_gate':'none_geometry_only',
        'near_plane':.001*prior.scale,'far_plane':100*prior.scale,
        'builder_implementation_sha256':sha256_file(__file__),
        'authority_implementation_sha256':sha256_file(authority_module.__file__),
        'compositor_implementation_sha256':sha256_file(inspect.getsourcefile(inspect.unwrap(rasterize_single_view_contributions))),
        'cameras':{}}
    for frame in frames:
        c=prior.cameras[frame];k=c.intrinsic.clone();k[0]*=args.width/c.width;k[1]*=args.height/c.height
        metadata['cameras'][str(frame)]={'intrinsic':k.tolist(),'world_to_camera':c.world_to_camera.tolist()}
    writer=SparseExactMarginalAuthorityWriter(out/'responsibility.json',metadata=metadata,frame_indices=frames,
        num_gaussians=len(model.xyz),num_pixels=args.width*args.height)
    for index,frame in enumerate(frames):
        c=metadata['cameras'][str(frame)];k=torch.tensor(c['intrinsic'])
        renderer=FeatureFieldRenderer(args.height,args.width,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]),
            near_plane=metadata['near_plane'],far_plane=metadata['far_plane']).to(args.device)
        hits=rasterize_single_view_contributions(model,renderer,torch.tensor(c['world_to_camera'],device=args.device),
            height=args.height,width=args.width)
        writer.add_view(index,hits['gaussian_ids'],hits['pixel_ids'],hits['weights'])
    authority,digest=writer.finalize();model.assert_binding(binding)
    records=[prior.records[f] for f in frames]
    image_dirs={str(Path(r['source_image']).parent) for r in records}
    if len(image_dirs)!=1:raise ValueError("source images must share a directory")
    source={'scene':prior.manifest.get('scene_label',''),'image_dir':image_dirs.pop(),'frames':[{'frame_idx':f,'source_file':Path(r['source_image']).name,
        'source_sha256':r['source_image_sha256']} for f,r in zip(frames,records)],
        'geometry_report_sha256':args.expected_report_sha256,'features_opened':False}
    (out/'source_frames.json').write_text(json.dumps(source,indent=2)+'\n')
    result={'authority':str(authority),'sha256':digest,'frames':frames,'feature_models_loaded':[]}
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--geometry-report',required=True);p.add_argument('--expected-report-sha256',required=True)
    p.add_argument('--frame-ids',required=True);p.add_argument('--height',type=int,default=46)
    p.add_argument('--width',type=int,default=62);p.add_argument('--device',default='cuda:0')
    p.add_argument('--output-dir',required=True)
    torch.set_num_threads(4);print(json.dumps(build(p.parse_args()),indent=2))
if __name__=='__main__':main()
