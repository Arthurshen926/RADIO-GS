"""Prepare missing legal query-free NVOS sources; no prompts, tracks or GT inputs."""
from pathlib import Path
import json
from scripts.research.run_query_field_relations import read,write,receipt,check

ROOT=Path('output/query_field_20261009/nvos_missing_sources_v1')
INVENTORY=Path('output/query_field_20261009/joint_observation_transfer_v1/diagnostics/remaining_family_sources.json')


def main():
    ROOT.mkdir(parents=True,exist_ok=False);inventory=read(INVENTORY);jobs=[]
    for row in inventory['records']:
        if row['family']!='nvos':continue
        scene_path=Path('configs/pipeline/nvos_all_sources')/(row['scene']+'.scene.json');scene=read(scene_path)
        if row['source_frames']!=[f for f in scene['source_frames'] if f not in row['development_excluded']]:
            raise ValueError('legal source roster changed')
        root=ROOT/row['scene'];rgb=root/'rgb';rgb.mkdir(parents=True)
        entries=[]
        for frame in row['missing_native_frames']:
            if frame not in row['source_frames']:raise ValueError('missing frame outside source authority')
            source=Path(scene['source_rgb'][str(frame)]).resolve(strict=True)
            target=rgb/(f'frame_{frame:05d}'+source.suffix.lower());target.symlink_to(source)
            entries.append(dict(frame=frame,rgb=str(source),rgb_sha256=receipt(source)['sha256'],staged=str(target)))
        source_cfg=read('output/query_field_20261009/confidence_mixture_functions_v1/confidence_mixture_function/cohort.json')
        # Execute only after this preparation; cache generation is intentionally
        # not represented as already complete by these commands or manifests.
        command=['/root/miniconda3/envs/sam3/bin/python','-m','radio_gs.scripts.build_sam3_automatic_mask_cache',
            '--image-root',str(rgb),'--output-root',str(root/'source_sam'),'--checkpoint-path',source_cfg['sam_checkpoint'],
            '--grid-size','12','--minimum-area-fraction','0.0001','--maximum-area-fraction','0.9']
        job=dict(scene=row['scene'],source_scene=receipt(scene_path),frames=entries,excluded_development=row['development_excluded'],
            full_legal_frames=row['source_frames'],existing_native_directories=row['native_directories'],
            sam_command=command,encoder=source_cfg['encoder'],source_masks_generated=False,native_descriptors_generated=False,
            target_query_or_tracks_read=False)
        write(root/'job.json',job);jobs.append(dict(scene=row['scene'],missing=len(entries),job=receipt(root/'job.json')))
    if sum(j['missing'] for j in jobs)!=123 or len(jobs)!=7:raise ValueError('missing cache roster changed')
    report=dict(jobs=jobs,scenes=7,missing_frames=123,source_only=True,queries_read=False,labels_read=False,
        build_complete=False,study_complete=False,source_cache_generation_pending=True,
        evidence=[receipt(INVENTORY),receipt(Path(__file__))])
    write(ROOT/'preparation.json',report);write('docs/experiments/2026-10-09-nvos-missing-source-preparation.json',report)
    print('Prepared123 missing legal frames in7 scenes;SAM/native generation NOT yet done.',flush=True)


if __name__=='__main__':main()
