"""Inventory reusable construction evidence in the other21 fields, without labels.

Presence and descriptor-header checks are preparation evidence, not completed
teacher/geometry transfer or proof that a mask payload is valid for reuse.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from scripts.research.run_query_field_joint_observation import ROOT,PREVIOUS,read,write,receipt


def main():
    prior=read(PREVIOUS/'design.json');legacy=read('configs/pipeline/system_v10.json')
    jobs=[j for j in prior['fields'] if j['family']!='uqis'];records=[]
    for job in jobs:
        family,name=job['family'],job['scene']
        old=next(j for j in legacy['jobs'] if j['benchmark']==family and j['name']==name)
        scene=read(old['scene']);source=read(Path(job['observations'])/'observations.json')
        development=source['splits']['development_frames'];frames=[f for f in scene['source_frames'] if f not in development]
        configs=[receipt(old['scene'])]
        if family=='scannet':
            path=Path('configs/consolidated/scannet_native')/(name+'.scene.json');native_scene=read(path);configs.append(receipt(path))
            directories=[Path(p) for p in native_scene['native_region_directories']]
        elif family=='lerf':directories=[Path(p) for p in scene['native_region_directories']]
        else:directories=sorted({Path(q['region_cache']['path']).parent for q in source['frames']})
        def inspect(frame):
            paths=[p/f'frame_{frame:05d}.pt' for p in directories]
            found=next((p for p in paths if p.exists()),None)
            record=dict(frame=frame,existing_native=None,query_independent_header=False,sam_exists=False)
            if found is not None:
                data=torch.load(found,map_location='cpu',weights_only=False)
                record.update(existing_native=receipt(found),frame_matches=int(data['frame'])==frame,
                    query_independent_header=data.get('query_independent') is True,
                    descriptor_shape=list(data['descriptors'].shape),sam_path=str(data['sam_path']),
                    sam_exists=Path(data['sam_path']).exists())
            elif family=='nvos':
                # This is only an available raw-mask candidate, not a native
                # teacher cache or an automatic query-independence certificate.
                image=Path(scene['source_rgb'][str(frame)])
                candidate=Path('output/nvos_source_sam_20260917')/name/(image.stem+'.pt')
                record.update(raw_sam_candidate=str(candidate),raw_sam_candidate_exists=candidate.exists())
            return record
        with ThreadPoolExecutor(4) as pool:views=list(pool.map(inspect,frames))
        current_geometry=source['geometry'];old_geometry=scene['geometry']
        records.append(dict(field_key=job['key'],family=family,scene=name,
            current_training_views=len(source['splits']['train_frames']),candidate_views=len(frames),
            development_excluded=development,source_frames=frames,
            same_geometry_receipt=current_geometry['sha256']==old_geometry['sha256'],
            current_geometry=current_geometry,v10_geometry=old_geometry,
            reusable_header_candidates=sum(v['existing_native'] is not None and v.get('frame_matches',False)
                and v['query_independent_header'] and v['sam_exists'] for v in views),
            missing_native_frames=[v['frame'] for v in views if v['existing_native'] is None],
            invalid_native_headers=[v['frame'] for v in views if v['existing_native'] is not None and
                not (v.get('frame_matches',False) and v['query_independent_header'] and v['sam_exists'])],
            native_directories=list(map(str,directories)),views=views,source_manifests=configs))
        print('source inventory',family,name,len(frames),records[-1]['reusable_header_candidates'],flush=True)
    summary={}
    for family in ['lerf','scannet','nvos']:
        rows=[r for r in records if r['family']==family]
        summary[family]=dict(fields=len(rows),same_geometry_receipts=sum(r['same_geometry_receipt'] for r in rows),
            **{k:sum(r[k] for r in rows) for k in ['current_training_views','candidate_views','reusable_header_candidates']},
            invalid_native_headers=sum(len(r['invalid_native_headers']) for r in rows))
    write(ROOT/'diagnostics/remaining_family_sources.json',dict(records=records,summary=summary,
        model_changed=False,benchmark_labels_accessed=False,benchmark_queries_accessed=False,
        function_transfer_complete=False,mask_payloads_rehashed=False,
        limitation='descriptor headers and referenced mask presence only; source authority, raster alignment and full function still require verification',
        code=receipt(Path(__file__)),prior_design=receipt(PREVIOUS/'design.json')))
    print(summary,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);main()
