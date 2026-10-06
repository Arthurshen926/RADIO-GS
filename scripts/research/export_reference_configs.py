"""Export the retained NVOS input contract; no predictions enter runtime configs."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    manifest=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text())
    out=Path('configs/consolidated/nvos');out.mkdir(parents=True,exist_ok=True)
    for item in manifest['scenes']:
        scene=item['scene_id'];map_scene='horns_center' if scene=='horns_left' else scene
        data=torch.load(Path('output/taskwise_scope_composition_next38')/f'nvos_{map_scene}'/'field.pt',map_location='cpu',weights_only=False)
        gr=json.loads(Path(data['geometry_report']).read_text());info=json.loads((Path(data['root'])/'teachers.json').read_text())
        report=json.loads(Path(f'output/nvos_reference_completion_next19/{scene}/report.json').read_text());receipt=json.loads((Path(report['query_root'])/scene/'receipt.json').read_text())
        ref=next(f for f in item['frames'] if f['frame_id']==item['prompt']['frame_id']);target=next(f for f in item['frames'] if f['frame_id'] in item['evaluation_frame_ids'])
        frames={int(r['rgb_sorted_index']):r['rgb_path'] for r in item['training_frames'] if r['frame_id'] not in item['evaluation_frame_ids'] and r['frame_id'] not in item['excluded_training_frame_ids']}
        frames[int(ref['canonical_index'])]=receipt['reference_rgb']['path'];tracking=sorted(frames)
        fitting=sorted((set(frames)-set(info['validation_frames']))|{int(ref['canonical_index'])})
        geometry=Path(f'output/nvos_carrier_next72/adaptive_allrgb/{map_scene}/geometry.pt')
        config=dict(scene=scene,geometry=dict(path=str(geometry),sha256=sha256_file(geometry)),camera_report=gr['source_report'],camera_report_sha256=gr['source_report_sha256'],source_frames=fitting,tracking_frames=tracking,source_rgb={str(f):frames[f] for f in tracking},source_fit_hw=[192,256],boundary_radius=2,checkpoint='checkpoints/sam3_modelscope/sam3.pt')
        query=dict(id=scene,modality='prompt_2d',intent='reference_instance',reference_frame=int(ref['canonical_index']),reference_rgb=receipt['reference_rgb']['path'],reference_mask=receipt['mask'],reference_mask_is_gt=False,positive_scribble=receipt['official_scribbles']['positive']['path'],negative_prompt_used=False,prompt_raster_wh=receipt['prompt_raster_wh'],seed_origin='retained positive-only SAM mask, not newly inferred mask')
        scoring=dict(scene=scene,target_frame=int(target['canonical_index']),gt_mask=target['gt_mask_path'],raster_wh=receipt['prompt_raster_wh'],threshold=.5,resize='bilinear continuous response before threshold',protocol='retained positive-only NVOS development; not signed official prompt reproduction')
        for name,value in [('scene',config),('query',query),('evaluation',scoring)]:
            (out/f'{scene}.{name}.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
