"""All legal non-reference NVOS source masks for the shared identity verifier."""
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
root=Path('output/reference_referent_verification');root.mkdir(parents=True,exist_ok=True)
jobs=[];sources={}
for scene in ['orchids','leaves','fern','flower','fortress','horns_center','horns_left','trex']:
    cfg=json.loads(Path(f'configs/consolidated/nvos/{scene}.scene.json').read_text())
    query=json.loads(Path(f'configs/consolidated/nvos/{scene}.query.json').read_text())
    original=json.loads(Path(f'output/reference_integration/replay/{scene}/fit.json').read_text())['tracks']
    tracks=torch.load(original,map_location='cpu',weights_only=False);sources[scene]=original
    out=root/'evidence'/scene;out.mkdir(parents=True,exist_ok=True)
    for frame in cfg['source_frames']:
        if frame==query['reference_frame']:continue
        record=tracks['frames'].get(frame)
        if record is None or record['object_logit']<=0:continue
        masks=record['logits'].float().sigmoid()[None]
        if not (masks>.5).any():continue
        path=out/f'identity_{frame}.pt';torch.save(dict(queries=[scene],instances=[dict(masks=masks.half(),scores=torch.ones(1))]),path)
        jobs.append(dict(key=f'{scene}:{frame}',scene=scene,query=scene,frame=frame,query_index=0,instance_index=0,
            evidence=str(path),source_rgb=cfg['source_rgb'][str(frame)],reference_rgb=query['reference_rgb'],reference_mask=query['reference_mask']))
(root/'manifest.json').write_text(json.dumps(dict(jobs=jobs,tracks=sources,reference_anchor_retained=True,
    target_rgb_or_gt_used=False,query_names_not_in_verifier_prompt=True,source_frame_cap=None),indent=2)+'\n')
print(len(jobs),'non-reference source regions')
