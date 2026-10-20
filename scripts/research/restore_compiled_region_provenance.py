"""Recover exact region/view lineage without rewriting any large field or map.

CPU-only. The main field is checked against every original descriptor row.
Replacement fields are checked against the parent's full sparse transfer and
recorded view/region order; their encoder is not rerun. No independent geometric
visibility is recovered. The derived recipe is opt-in and retains all 31 jobs.
"""
import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import time

import torch

from radio_gs.evidence_core.provenance import (file_sha256, field_binding,
    verify_native_block, verify_replacement_lineage, attach_provenance)
from radio_gs.evidence_core.store import CompiledRegionEvidence, region_layout
from radio_gs.querying.native_observation_prior import compiled_observation_prior


@contextmanager
def shared_profile():
    old=os.environ.get('RADIO_GS_EVIDENCE_CORE')
    os.environ['RADIO_GS_EVIDENCE_CORE']='shared_v1'
    try:
        yield
    finally:
        if old is None:os.environ.pop('RADIO_GS_EVIDENCE_CORE',None)
        else:os.environ['RADIO_GS_EVIDENCE_CORE']=old


def read(path):return json.loads(Path(path).read_text())
def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def write(path, value):
    data=(json.dumps(value,indent=2)+'\n').encode()
    if len(data)>8*2**20:raise ValueError('metadata artifact exceeds 8 MiB; no truncation permitted')
    with Path(path).open('xb') as stream:stream.write(data)


def seal(path):return dict(path=str(Path(path).resolve()),sha256=file_sha256(path))


def check_readout(scene, payload, reference):
    query=torch.eye(19,payload['descriptors'].shape[1])
    fallback=torch.zeros(payload['shape'][0],len(query))
    with shared_profile():
        before=compiled_observation_prior(scene,query,payload['supported'],fallback,field=payload)[0]
        after=compiled_observation_prior(dict(scene,native_observation_provenance=reference),
            query,payload['supported'],fallback,field=payload)[0]
    if not torch.equal(before,after):raise AssertionError('provenance overlay changed readout')
    return dict(exact=True,entries=before.numel(),max_abs_error=0.,
                query='fixed orthogonal fixture, not teacher-encoded benchmark classes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',default='configs/pipeline/system_v10.json')
    parser.add_argument('--native-scenes',default='configs/pipeline/scannet_native_complete')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();torch.set_num_threads(4);started=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    recipe=read(args.recipe);derived=copy.deepcopy(recipe);results=[]
    sources=[Path(__file__),Path('radio_gs/evidence_core/provenance.py'),Path('radio_gs/evidence_core/store.py'),
             Path('radio_gs/querying/native_observation_prior.py'),Path('radio_gs/pipeline/observation_field.py'),
             Path('radio_gs/pipeline/adapter_observation_field.py')]
    source_hashes={str(p):file_sha256(p) for p in sources}
    for job in derived['jobs']:
        if job['benchmark']!='scannet':continue
        scene=read(job['scene']);native_path=Path(args.native_scenes)/Path(job['scene']).name
        native=read(native_path);parent_path=Path(scene['native_observation_field']);parent=load(parent_path)
        if scene['source_frames']!=native['source_frames'] or scene['geometry']!=native['geometry'] or parent['geometry']!=scene['geometry']:
            raise ValueError('scene/native geometry or frame order differs')
        offsets=[0];records=[]
        for frame in native['source_frames']:
            path=Path(native['native_region_directories'][0])/f'frame_{frame:05d}.pt'
            binding=seal(path);row=load(path)
            indices=verify_native_block(parent,row,frame,offsets[-1])
            if file_sha256(path)!=binding['sha256']:raise ValueError('cache changed during verification')
            offsets.append(offsets[-1]+len(indices))
            records.append(dict(frame=frame,proposal_indices=indices,cache=binding,
                                sam_path=row['sam_path'],sam_content_verified=False))
        region_layout(offsets,scene['source_frames'],parent['shape'][1])
        parent_binding=field_binding(parent)
        parent_seal=seal(parent_path)
        # No tensor duplication: the replacement is loaded sequentially once per room.
        aux_path=Path(scene['local_auxiliary_scene']);aux_scene=read(aux_path)
        aux_field_path=Path(aux_scene['native_observation_field']);aux_field=load(aux_field_path)
        build_path=aux_field_path.parent.parent/'build.json';build=read(build_path)
        if (Path(build['prior_field']).resolve()!=parent_path.resolve() or
                aux_scene['source_frames']!=scene['source_frames'] or not build['geometry_weights_unchanged']):
            raise ValueError('replacement build lineage differs')
        verify_replacement_lineage(parent,aux_field,build['records'],scene['source_frames'],offsets)
        scene_dir=args.output/job['name'];scene_dir.mkdir()
        specs=[]
        for kind,payload,path,spec in [('native',parent,parent_path,scene),('replacement',aux_field,aux_field_path,aux_scene)]:
            metadata=dict(schema='radiogs.compiled_region_provenance.v1',
                field=dict(**seal(path),binding=field_binding(payload)),
                source_frames=scene['source_frames'],region_offsets=offsets,regions_by_view=records,
                verification=('all_original_descriptor_rows_exact' if kind=='native' else
                              'parent_operator_exact_and_recorded_view_order'),
                parent_field=dict(**parent_seal,binding=parent_binding),
                source_scene=seal(native_path),configured_scene=seal(job['scene'] if kind=='native' else aux_path),
                replacement_build=seal(build_path) if kind=='replacement' else None,
                independent_visibility_recovered=False,replacement_encoder_replayed=False,
                source_mask_pixels_replayed=False,build_source_sha256=source_hashes)
            sidecar=scene_dir/(kind+'.provenance.json');write(sidecar,metadata);reference=seal(sidecar)
            overlay=attach_provenance(payload,path,reference,source_frames=spec['source_frames'])
            adapter=CompiledRegionEvidence.from_payload(overlay)
            if adapter.independent_visibility_available:raise AssertionError('legacy visibility unexpectedly available')
            probe=dict(spec)
            if kind=='replacement':
                probe['native_observation_reliability']=scene['native_observation_reliability']
            parity=check_readout(probe,payload,reference)
            derived_scene=dict(spec,native_observation_provenance=reference)
            target=scene_dir/(kind+'.scene.json');specs.append((target,derived_scene))
            results.append(dict(scene=job['name'],teacher=kind,field=str(path),reference=reference,
                views=len(records),regions=offsets[-1],provenance_available=adapter.region_frames is not None,
                independent_visibility_available=False,readout_check=parity,verification=metadata['verification']))
        specs[0][1]['local_auxiliary_scene']=str(specs[1][0].resolve())
        for path,spec in specs:write(path,spec)
        job['scene']=str(specs[0][0].resolve())
        print(job['name'],len(records),'views',offsets[-1],'regions, both teachers verified',flush=True)
        del parent,aux_field,payload,overlay,adapter
    if source_hashes!={str(p):file_sha256(p) for p in sources}:raise ValueError('source changed during recovery')
    write(args.output/'recipe.json',derived)
    report=dict(schema='radiogs.compiled_region_recovery.v1',recipe=seal(args.recipe),
        derived_recipe=seal(args.output/'recipe.json'),fields=len(results),rows=results,
        source_sha256=source_hashes,sources_unchanged_during_audit=True,seconds=time.monotonic()-started,
        full_benchmark_evaluation=False,ground_truth_read=False,source_rgb_read=False,
        source_masks_read=False,new_teacher_calls=0,large_assets_rewritten=False,
        all_jobs_retained=len(derived['jobs']),visibility_recovered=False,
        limits='Region identity recovery; auxiliary encoder and original mask pixels not replayed. No new geometric visibility or benchmark scores.')
    write(args.output/'report.json',report)
    size=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    if size>32*2**20:raise ValueError('recovery metadata exceeds 32 MiB; do not promote')
    print(json.dumps(dict(fields=len(results),durable_bytes=size,output=str(args.output))),flush=True)


if __name__=='__main__':main()
