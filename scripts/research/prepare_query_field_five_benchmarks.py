"""Prepare the full fixed architecture cohort; no benchmark label loading."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from radio_gs.benchmarks.uqis.sensor import SensorIndex
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.geometry.registered_mesh_carrier import build_registered_mesh_carrier
from radio_gs.query_field.storage import sha256


ROOT = Path('output/query_field_20261007/five_benchmark_v1')
CONFIG = Path('configs/query_field/five_benchmark_v1')
PACKAGE = Path('output/scannet_uqis_9_v0_3_1_candidate')
CHECKPOINT = '/root/RADIO-GS/checkpoints/sam3_modelscope/sam3.pt'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def select_frames(frames):
    frames = sorted(set(map(int, frames)))
    ids = np.linspace(0, len(frames)-1, min(20, len(frames))).round().astype(int)
    selected = [frames[i] for i in ids]
    development = [selected[len(selected)//4], selected[3*len(selected)//4]]
    return selected, [f for f in selected if f not in development], development


def camera_record(frame, camera):
    return dict(frame_index=frame, width=camera.width, height=camera.height,
                intrinsic=camera.intrinsic.tolist(), world_to_camera=camera.world_to_camera.tolist(),
                camera_to_world=torch.linalg.inv(camera.world_to_camera.double()).tolist())


def source_job(family, name, geometry, rows, images, frames, train, dev, encoder,
               *, scale=1., regions=None, geometry_condition='provided unlabelled mesh converted to frozen surface Gaussians'):
    root = ROOT/family/name
    root.mkdir(parents=True, exist_ok=True)
    authority = root/'source_authority.json'
    write(authority, dict(images=[dict(image_id=f'frame_{f:05d}',path=images[f],sha256=sha256(images[f])) for f in frames],
                         benchmark_queries_used=False, mapping_only=True))
    cameras = root/'cameras.json'
    write(cameras, dict(schema='registered_pinhole_source_authority.v1',
                        camera_convention='OpenCV: x right, y down, z forward',
                        records=rows, source_frame_indices=frames))
    scene_path = root/'scene.json'
    write(scene_path, dict(scene=name, geometry=geometry, registered_cameras=str(cameras),
          registered_cameras_sha256=sha256(cameras), world_units_per_metre=scale,
          source_frames=frames, source_rgb_authority=str(authority), text_encoder=encoder,
          native_dimension=1536, native_region_directories=[str(regions or root/'native_regions')],
          geometry_condition=geometry_condition))
    config = json.loads(Path('configs/query_field/figurines_spatial_metric.json').read_text())
    config.update(scene=str(scene_path), train_frames=train, development_frames=dev,
                  purpose='Full five-benchmark architecture baseline, no local mechanism sweep',
                  selection='Uniform sorted legal mapping frames; two fixed quartile views excluded from feature fitting',
                  input_budget_condition=f'{len(train)} training and {len(dev)} teacher-development views; saved geometry track')
    config_path = CONFIG/f'{family}_{name}.json'
    write(config_path, config)
    return dict(family=family, name=name, output=str(root), config=str(config_path),
                field=str(root/'models/trained'), needs_sam=regions is None,
                images=[dict(frame=f,rgb=images[f],rgb_sha256=sha256(images[f])) for f in frames],
                train_frames=train, development_frames=dev, geometry=geometry)


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    CONFIG.mkdir(parents=True, exist_ok=False)
    encoder=json.loads(Path('output/system_map_rebuild/figurines/complete/scene.json').read_text())['text_encoder']
    baseline=json.loads(Path('output/query_field_20261007/frozen_v10/baseline_a.json').read_text())
    jobs, nvos = [], []
    # Existing four LERF fields/predictions and fern remain the same architecture.
    references=dict(figurines='spatial_metric_lerf_v1/figurines_fitted_metric',
                    ramen='spatial_metric_lerf_v1/ramen_fitted_metric',
                    teatime='spatial_metric_teatime_lerf_v1/teatime_fitted_metric',
                    waldo_kitchen='waldo_kitchen_physical_lerf_v1/waldo_kitchen')
    lerf=[dict(scene=s, field=f'output/query_field_20261007/{s}_fitted_metric_v1/models/trained',
               predictions='output/query_field_20261007/'+p) for s,p in references.items()]
    old_nvos=json.loads(Path('configs/query_field/nvos_fern_prediction.json').read_text())
    for path in sorted(Path('configs/pipeline/nvos_all_sources').glob('*.scene.json')):
        old=json.loads(path.read_text()); name=old['scene']
        query=json.loads(Path(f'configs/consolidated/nvos/{name}.query.json').read_text())
        evaluation=json.loads(Path(f'configs/consolidated/nvos/{name}.evaluation.json').read_text())
        if evaluation['target_frame'] in old['source_frames']:
            raise ValueError('NVOS target frame in construction authority')
        cameras=MogeGeometryPrior(old['camera_report'], old['camera_report_sha256'])
        canonical='horns_center' if name=='horns_left' else name
        root=ROOT/'nvos'/canonical
        if name=='fern':
            field='output/query_field_20261007/nvos_fern_v1/models/trained'
        else:
            field=str(root/'models/trained')
        if name not in {'fern','horns_left'}:
            frames,train,dev=select_frames(old['source_frames'])
            rgbroot=root/'rgb'; rgbroot.mkdir(parents=True)
            images={}
            for f in frames:
                image=Image.open(old['source_rgb'][str(f)]).convert('RGB')
                image.thumbnail((1008,1008),Image.Resampling.LANCZOS)
                dest=rgbroot/f'frame_{f:05d}.png';image.save(dest);images[f]=str(dest)
            job=source_job('nvos',name,old['geometry'],[camera_record(f,cameras.cameras[f]) for f in frames],
                           images,frames,train,dev,encoder,scale=cameras.scale,
                           geometry_condition='same frozen v10 NVOS carrier; source RGB resized to longest side1008')
            jobs.append(job)
        fixture={}
        for role,frame in [('reference',query['reference_frame']),('target',evaluation['target_frame'])]:
            camera=cameras.cameras[frame]
            fixture[role]=dict(intrinsic=camera.intrinsic,world_to_camera=camera.world_to_camera,
                               height=camera.height,width=camera.width,frame=frame)
            if role=='reference':
                width,height=query['prompt_raster_wh'];k=camera.intrinsic.clone()
                k[0]*=width/camera.width;k[1]*=height/camera.height
                fixture[role].update(intrinsic=k,height=height,width=width)
        fixture['world_units_per_metre']=cameras.scale
        fixture_path=ROOT/'nvos'/name/'query_cameras.pt';fixture_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(fixture,fixture_path)
        positive=Path(query['positive_scribble']);negative=positive.with_name(positive.name.replace('pos_','neg_',1))
        prediction={**old_nvos,'scene':name,'jobs':[dict(name=name,field=field,mode='signed',scope='default',
                    positive=str(positive),negative=str(negative),camera_fixture=str(fixture_path))]}
        prediction.update(evaluation_status='complete fixed eight-case development cohort',
                          prediction_policy='one fitted physical member field; signed raw prompts, default scale')
        pp=CONFIG/f'nvos_{name}_prediction.json';write(pp,prediction)
        sp=CONFIG/f'nvos_{name}_scoring.json';write(sp,dict(scene=name,gt_mask=evaluation['gt_mask'],prediction_config=str(pp)))
        nvos.append(dict(name=name,field=field,prediction_config=str(pp),scoring_config=str(sp),
                         reference_prediction=None if name!='fern' else 'output/query_field_20261007/nvos_fern_v1/public/trained_signed_default'))
    scannet=[]
    for item in baseline['jobs']:
        if item['job']['benchmark']!='scannet':continue
        name=item['job']['name'];old=json.loads(Path(f'configs/consolidated/scannet_native/{name}.scene.json').read_text())
        frames,train,dev=select_frames(old['source_frames'])
        table=json.loads(Path(old['source_cameras']).read_text());by_frame={r['frame_index']:r for r in table['records']}
        root=ROOT/'scannet'/name;regions=root/'native_regions';regions.mkdir(parents=True)
        images={f:by_frame[f]['source_image'] for f in frames}
        # Reuse native source descriptors, filtering by the same construction area rule.
        for f in frames:
            source=Path(old['native_region_directories'][0])/f'frame_{f:05d}.pt'
            region=torch.load(source,map_location='cpu',weights_only=False)
            sam=torch.load(region['sam_path'],map_location='cpu',weights_only=False)
            area=sam['proposal_area_fraction'][region['native_proposal_indices']]
            keep=(area>=.002)&(area<=.6)
            if not keep.any():raise ValueError('no eligible source masks')
            region['descriptors']=region['descriptors'][keep];region['native_proposal_indices']=region['native_proposal_indices'][keep]
            region['source_descriptor_cache']=dict(path=str(source),sha256=sha256(source))
            torch.save(region,regions/f'frame_{f:05d}.pt')
        jobs.append(source_job('scannet',name,old['geometry'],[by_frame[f] for f in frames],images,
                               frames,train,dev,encoder,regions=regions))
        scannet.append(dict(name=name,field=str(root/'models/trained'),queries=item['job']['queries'],evaluation=item['job']['evaluation']))
    mapping=json.loads((PACKAGE/'mapping/scenes.json').read_text())['scenes']
    domains=json.loads((PACKAGE/'method/scenes.json').read_text())['scene_domains']
    uqis=[]
    for source in mapping:
        name=source['scene_id'];domain=next(d for d in domains if d['scene_id']==name)
        frames,train,dev=select_frames([f for f in source['field_frame_ids'] if int(f)%20==0])
        if set(frames)&set(map(int,source['withheld_frame_ids'])):raise ValueError('UQIS excluded frame selected')
        root=ROOT/'uqis'/name;rgbroot=root/'rgb';rgbroot.mkdir(parents=True)
        sensor=SensorIndex(source['sens_path']);images={};rows=[]
        for f in frames:
            pose,_,image=sensor.read(f);width=640;height=round(image.height*width/image.width)
            image=image.resize((width,height),Image.Resampling.LANCZOS);dest=rgbroot/f'frame_{f:05d}.png';image.save(dest)
            images[f]=str(dest);k=sensor.k_color[:3,:3].copy();k[0]*=width/sensor.color_width;k[1]*=height/sensor.color_height
            rows.append(dict(frame_index=f,width=width,height=height,intrinsic=k.tolist(),
                             camera_to_world=pose.tolist(),world_to_camera=np.linalg.inv(pose).tolist()))
        mesh=Path('/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small/annotations')/name/f'{name}_vh_clean_2.ply'
        ply=PlyData.read(mesh);vertices=np.stack([ply['vertex'][k] for k in ['x','y','z']],1)
        public=PACKAGE/'method'/domain['mesh_xyz_path']
        if sha256(public)!=domain['mesh_xyz_sha256'] or not np.array_equal(vertices,np.load(public)):
            raise ValueError('UQIS public geometry differs from unlabelled construction mesh')
        model,ids=build_registered_mesh_carrier(vertices,np.stack(ply['face']['vertex_indices']))
        geometry=root/'geometry.pt'
        torch.save(dict(schema='radio_gs.frozen_surface_gaussian_geometry.v1',geometry_trainable_during_semantics=False,
                        state_dict=model.state_dict(),input_track='official_unlabelled_mesh_Gaussian_surface',
                        source_vertex_ids=torch.from_numpy(ids)),geometry)
        jobs.append(source_job('uqis',name,dict(path=str(geometry),sha256=sha256(geometry)),rows,images,
                               frames,train,dev,encoder))
        uqis.append(dict(name=name,field=str(root/'models/trained'),public_domain=domain,
                         geometry=dict(path=str(geometry),sha256=sha256(geometry)),
                         mapping_manifest_sha256=sha256(PACKAGE/'mapping/scenes.json')))
    write(CONFIG/'cohort.json',dict(parent_commit='fb3c768',root=str(ROOT),construction=jobs,
          lerf=lerf,nvos=nvos,scannet=scannet,uqis=uqis,uqis_package=str(PACKAGE),encoder=encoder,
          sam_checkpoint=CHECKPOINT,shared_training_recipe='figurines_spatial_metric',
          source_budget='18 train /2 development uniformly sampled legal views for new fields; retained LERF18 and fern12',
          primary_readouts=dict(lerf2d='shared_membership',lerf3d='actual_subset_projection',scannet='shared_member_category_competition',
                                nvos='registered_signed_membership',uqis='shared_member_instance'),
          independent_final_test=False,parameter_search=False))
    print('prepared',len(jobs),'new fields;4 LERF scenes,8 NVOS cases,10 ScanNet rooms,9 UQIS rooms',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
