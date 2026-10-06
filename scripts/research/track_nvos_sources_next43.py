"""Reference-initialized SAM3 video tracking over legal map-source RGB only."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from sam3.model_builder import build_sam3_video_model


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--scene-input');p.add_argument('--query');p.add_argument('--all-legal-sources',action='store_true');p.add_argument('--output',default='output/source_video_next43');a=p.parse_args();torch.set_num_threads(4)
    if a.scene_input:
        config=json.loads(Path(a.scene_input).read_text());query=json.loads(Path(a.query).read_text())
        frames={f:config['source_rgb'][str(f)] for f in config['tracking_frames']}
        ref=query['reference_frame'];order=sorted(frames);seed_index=order.index(ref)
        qr=dict(mask=query.get('reference_mask'),reference_rgb=dict(path=query['reference_rgb']))
    else:
        key='nvos_'+('horns_center' if a.scene=='horns_left' else a.scene)
        data=torch.load(Path('output/taskwise_scope_composition_next38')/key/'field.pt',map_location='cpu',weights_only=False);root=Path(data['root'])
        source=json.loads((root/'source_frames.json').read_text());info=json.loads((root/'teachers.json').read_text())
        report=json.loads((Path('output/nvos_reference_completion_next19')/a.scene/'report.json').read_text());qr=json.loads((Path(report['query_root'])/a.scene/'receipt.json').read_text())
        manifest=json.loads(Path('/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json').read_text());scene=next(s for s in manifest['scenes'] if s['scene_id']==a.scene)
        reference=next(f for f in scene['frames'] if f['frame_id']==scene['prompt']['frame_id']);forbidden={f['canonical_index'] for f in scene['frames'] if f['frame_id'] in scene['evaluation_frame_ids']}
        frames={int(r['frame_idx']):str(Path(source['image_dir'])/r['source_file']) for r in source['frames'] if r['frame_idx'] in info['train_frames'] and r['frame_idx'] not in forbidden}
        if a.all_legal_sources:
            frames={int(r['rgb_sorted_index']):r['rgb_path'] for r in scene['training_frames'] if r['frame_id'] not in scene['evaluation_frame_ids'] and r['frame_id'] not in scene['excluded_training_frame_ids']}
        ref=int(reference['canonical_index']);frames[ref]=qr['reference_rgb']['path'];order=sorted(frames);seed_index=order.index(ref)
    points=query.get('points_normalized') if a.scene_input else None
    if points is None:
        seed=np.asarray(Image.open(qr['mask']));seed=np.any(seed>127,axis=2) if seed.ndim==3 else seed>127
    out=Path(a.output)/a.scene;out.mkdir(parents=True,exist_ok=True)
    model=build_sam3_video_model(checkpoint_path=config['checkpoint'] if a.scene_input else 'checkpoints/sam3_modelscope/sam3.pt',load_from_HF=False,strict_state_dict_loading=True,apply_temporal_disambiguation=True,device='cuda',compile=False)
    tracker=model.tracker;tracker.backbone=model.detector.backbone;predictions={}
    with tempfile.TemporaryDirectory(prefix='radio_source_video_') as temp,torch.autocast('cuda',dtype=torch.bfloat16):
        # One common aspect-preserving source raster, same registered framing.
        size=Image.open(qr['reference_rgb']['path']).size;width=1008;height=round(size[1]*width/size[0])
        for i,frame in enumerate(order):Image.open(frames[frame]).convert('RGB').resize((width,height),Image.Resampling.BILINEAR).save(Path(temp)/f'{i:05d}.jpg')
        state=tracker.init_state(video_path=temp,offload_video_to_cpu=True)
        if points is None:
            tracker.add_new_mask(inference_state=state,frame_idx=seed_index,obj_id=1,mask=torch.from_numpy(seed))
        else:
            tracker.add_new_points_or_box(inference_state=state,frame_idx=seed_index,obj_id=1,points=torch.tensor(points,dtype=torch.float32),labels=torch.tensor(query['point_labels'],dtype=torch.int32),rel_coordinates=True)
        for reverse in [False,True]:
            for index,ids,_low,masks,scores in tracker.propagate_in_video(state,start_frame_idx=seed_index,max_frame_num_to_track=len(order),reverse=reverse,propagate_preflight=True):
                k=list(ids).index(1);frame=order[int(index)]
                predictions[frame]=dict(logits=masks[k].float().cpu().squeeze().half(),object_logit=float(torch.as_tensor(scores[k]).cpu()))
    torch.save(dict(frames=predictions,reference_frame=ref),out/'tracks.pt')
    (out/'tracking.json').write_text(json.dumps(dict(explicit_scene=a.scene_input,explicit_query=a.query,scene=a.scene,source_frames=order,source_rgb=frames,reference_frame=ref,target_rgb_used=False,negative_prompt_used=False,teacher='SAM3 temporal tracker',all_legal_sources=a.all_legal_sources,seed='existing positive-only reference mask' if points is None else 'explicit normalized point prompts',missing_object_is_unknown=True),indent=2)+'\n');print(a.scene,'tracked',len(predictions),flush=True)


if __name__=='__main__':main()
