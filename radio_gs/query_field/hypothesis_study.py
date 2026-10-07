"""Stage complete predictions before evaluating finite candidate capacity."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .contracts import QueryPacket,Support
from .evaluate import choose_pixel,mask_iou
from .geometry import selected_alpha
from .hypotheses import propose,choose
from .observations import erode
from .storage import load_field,sha256


def mean(rows,key):
    return sum(r[key] for r in rows)/len(rows)


@torch.no_grad()
def run(config_path,output,device='cuda'):
    config=json.loads(Path(config_path).read_text())
    if not config['scale_factors'] or config['scale_factors'][0]!=1.:
        raise ValueError('the first scale must replay the original default')
    if set(config['automatic_controls'])!={'anchor','native_bidirectional'}:
        raise ValueError('report requires the declared anchor and native controls')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    field,export=load_field(config['field'],device)
    if field.candidate_count!=config['candidate_count']:
        raise ValueError('fixed candidate budget differs from the scene package')
    root=Path(config['observations'])
    manifest=json.loads((root/'observations.json').read_text())
    height,width=manifest['raster_hw']
    staged=[];views={}
    for entry in manifest['frames']:
        if entry['split']!=config['split']:continue
        path=root/f"view_{entry['frame']:05d}.pt"
        if sha256(path)!=entry['compiled']['sha256']:raise ValueError('observation changed')
        view=torch.load(path,map_location='cpu',weights_only=True)
        views[entry['frame']]=view
        camera=SimpleNamespace(intrinsic=view['intrinsic'],world_to_camera=view['world_to_camera'],
            height=view['original_hw'][0],width=view['original_hw'][1])
        for index,mask in enumerate(view['masks']):
            inside=torch.where(erode(mask).flatten())[0]
            if not len(inside):inside=torch.where(mask.flatten())[0]
            row=dict(frame=entry['frame'],region=index,status='ok',hypotheses=[],point_scales=[])
            if not len(inside):
                row['status']='empty_fixed_raster_mask';staged.append(row);continue
            positions=torch.linspace(0,len(inside)-1,min(16,len(inside))).round().long()
            packet=QueryPacket('image',semantic=view['semantic_descriptors'][index].to(device),
                               local=view['local_pixels'][inside[positions]].to(device))
            hypotheses=propose(field,packet,config['scale_factors'])
            masks=[]
            for h in hypotheses:
                alpha=selected_alpha(field,h.selected,camera,height,width,manifest['world_units_per_metre'])
                masks.append((alpha>.5).cpu())
                row['hypotheses'].append(dict(anchor=h.anchor,scale=h.scale,anchor_score=h.anchor_score,
                                                native_score=h.native_score,evidence=h.evidence))
            row['predicted_masks']=torch.stack(masks)
            row['automatic_selection']={policy:choose(hypotheses,policy) for policy in config['automatic_controls']}
            # The point association is input construction, not answer-driven
            # selection within the text/image hypothesis pool.
            pixel=choose_pixel(mask);anchor=int(view['best_element'][pixel])
            if anchor>=0 and bool(field.observed[anchor]):
                packet=QueryPacket('point_3d',positive=Support(torch.tensor([anchor],device=device),
                                   torch.ones(1,device=device)),exact_positive=True)
                point_hypotheses=propose(field,packet,config['scale_factors'])
                row['point_predicted_masks']=torch.stack([
                    (selected_alpha(field,h.selected,camera,height,width,manifest['world_units_per_metre'])>.5).cpu()
                    for h in point_hypotheses])
                row['point_scales']=[h.scale for h in point_hypotheses]
            staged.append(row)
            if index%20==0:print('staged hypotheses',entry['frame'],index,flush=True)
    predictions=output/'predictions.pt';torch.save(staged,predictions)
    # No teacher-target IoU is calculated before this complete materialization.
    rows=[]
    for staged_row in staged:
        row={k:v for k,v in staged_row.items() if k not in {'predicted_masks','point_predicted_masks'}}
        row.update(anchor_iou=0.,native_bidirectional_iou=0.,oracle_identity_iou=0.,
            oracle_scale_at_anchor_iou=0.,oracle_identity_scale_iou=0.,point_default_iou=0.,point_oracle_scale_iou=0.)
        if staged_row['status']=='ok':
            target=views[row['frame']]['masks'][row['region']].flatten()
            ious=[mask_iou(mask,target) for mask in staged_row['predicted_masks']]
            count=len(ious)//len(config['scale_factors'])
            row['hypothesis_ious']=ious
            row['anchor_iou']=ious[staged_row['automatic_selection']['anchor']]
            row['native_bidirectional_iou']=ious[staged_row['automatic_selection']['native_bidirectional']]
            row['oracle_identity_iou']=max(ious[:count])
            row['oracle_scale_at_anchor_iou']=max(ious[::count])
            row['oracle_identity_scale_iou']=max(ious)
            if 'point_predicted_masks' in staged_row:
                point_ious=[mask_iou(mask,target) for mask in staged_row['point_predicted_masks']]
                row['point_default_iou']=point_ious[0];row['point_oracle_scale_iou']=max(point_ious)
        rows.append(row)
    keys=['anchor_iou','native_bidirectional_iou','oracle_identity_iou','oracle_scale_at_anchor_iou',
          'oracle_identity_scale_iou','point_default_iou','point_oracle_scale_iou']
    report=dict(schema='radio_gs.member_hypothesis_study.v1',config=config,configuration_sha256=sha256(config_path),
        field_sha256=export['files']['field.pt']['sha256'],observations_sha256=sha256(root/'observations.json'),
        predictions_sha256=sha256(predictions),all_predictions_before_target_iou=True,cases=len(rows),
        benchmark_result=False,independent_test=False,query_contract='teacher-region image diagnostic; not external crop benchmark',
        privileged_metrics=[key for key in keys if 'oracle' in key],
        oracle_is_not_deployed=True,point_unsupported=sum(not r['point_scales'] for r in rows),
        summary={key:mean(rows,key) for key in keys},rows=rows,promotion=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return {key:value for key,value in report.items() if key!='rows'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(run(args.config,args.output,args.device),indent=2))


if __name__=='__main__':main()
