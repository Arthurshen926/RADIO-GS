"""Source-only descriptor diagnostics; never independent identity ground truth.

Match each tracked mask to a query-independent source region by mask IoU, then
measure native text and seed appearance similarity. No benchmark masks are read.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);results=[]
    for scene in ['figurines','ramen','teatime','waldo_kitchen']:
        root=Path('output/research_consolidation/regional_run')/scene
        manifest=json.loads((root/'seeds/seeds.json').read_text())
        config=json.loads(Path(f'configs/consolidated/{scene}.scene.json').read_text())
        inputs=json.loads((root/'inputs.json').read_text());response=load(inputs['response'])
        query=F.normalize(response['encodings'].float(),dim=-1)
        tracks={int(q):load(root/'tracks'/f'lerf_{scene}'/f'{q}.pt') for q in manifest['seeds']}
        seeds={int(q):load(path) for q,path in manifest['seeds'].items()}
        seed_features={q:F.normalize(load(s['candidate']['path'])['descriptors'][s['candidate']['index']].float(),dim=0) for q,s in seeds.items()}
        rows={q:[] for q in seeds}
        for frame in config['source_frames']:
            active={q:t['frames'][frame] for q,t in tracks.items() if frame in t['frames'] and t['frames'][frame]['present']}
            if not active:continue
            desc=[];masks=[]
            for directory in config['native_region_directories']:
                native=load(Path(directory)/f'frame_{frame:05d}.pt')
                if not len(native['descriptors']):continue
                sam=load(native['sam_path']);unpacked=unpack_masks(sam['packed_masks'],sam['mask_shape'][1])
                chosen=unpacked[native['native_proposal_indices'].numpy()]
                masks.append(F.interpolate(torch.from_numpy(chosen).float()[:,None],size=(96,128),mode='nearest')[:,0].flatten(1).bool())
                desc.append(F.normalize(native['descriptors'].float(),dim=-1))
            if not masks:continue
            masks=torch.cat(masks);desc=torch.cat(desc);text_scores=desc@query.T
            for q,t in active.items():
                track=F.interpolate(t['probability'].float()[None,None],size=(96,128),mode='bilinear',align_corners=False)[0,0].flatten()>.5
                intersection=(masks&track).sum(1);union=(masks|track).sum(1)
                iou=intersection/union.clamp_min(1);index=int(iou.argmax());scores=text_scores[index]
                rows[q].append(dict(frame=frame,region_track_iou=float(iou[index]),text_cosine=float(scores[q]),seed_cosine=float(desc[index]@seed_features[q]),query_rank=int((scores>scores[q]).sum()+1),top_text=response['texts'][int(scores.argmax())]))
        for q,items in rows.items():
            matched=[r for r in items if r['region_track_iou']>=.5]
            s=seed_features[q]@query.T
            results.append(dict(scene=scene,query=response['texts'][q],query_index=q,seed_frame=seeds[q]['seed_frame'],seed_native_score=float(s[q]),seed_query_rank=int((s>s[q]).sum()+1),seed_top_text=response['texts'][int(s.argmax())],present_views=len(items),matched_views=len(matched),matched_mean_text_cosine=float(np.mean([r['text_cosine'] for r in matched])) if matched else None,matched_mean_seed_cosine=float(np.mean([r['seed_cosine'] for r in matched])) if matched else None,frames=items))
        print(scene,'complete',flush=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(dict(independent_truth=False,heldout_validation=False,policy='mask-IoU correspondence to existing native region dictionary; text ranking within current query list is diagnostic only',queries=results),indent=2)+'\n')


if __name__=='__main__':main()
