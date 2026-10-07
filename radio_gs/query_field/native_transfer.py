"""Matched native-coordinate intervention with fixed grouping and support."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .contracts import QueryPacket
from .evaluate import mask_iou
from .geometry import selected_alpha
from .observations import erode
from .storage import load_field,sha256


@torch.no_grad()
def run(config_path,output,device='cuda'):
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    earlier,old_export=load_field(config['earlier_field'],device)
    later,new_export=load_field(config['later_field'],device)
    geometry=['xyz','normals','quaternions','scales','opacities']
    for key in geometry:
        if not torch.equal(getattr(earlier,key),getattr(later,key)):raise ValueError('geometry differs')
    common=earlier.observed&later.observed
    later.observed.copy_(common)
    channels={'six_view':(earlier.semantic,earlier.local),'eighteen_view':(later.semantic,later.local)}
    protected={key:value.clone() for key,value in later.state_dict().items() if not key.startswith(('semantic.','local.'))}
    old_root=Path(config['earlier_observations']);root=Path(config['later_observations'])
    old_manifest=json.loads((old_root/'observations.json').read_text());manifest=json.loads((root/'observations.json').read_text())
    height,width=manifest['raster_hw'];rows=[]
    for entry in manifest['frames']:
        if entry['split']!=config['split']:continue
        old_entry=next(row for row in old_manifest['frames'] if row['frame']==entry['frame'])
        if old_entry['split']!=config['split']:raise ValueError('the comparison split differs')
        path=root/f"view_{entry['frame']:05d}.pt";old_path=old_root/path.name
        if sha256(path)!=entry['compiled']['sha256'] or sha256(old_path)!=old_entry['compiled']['sha256']:
            raise ValueError('observation digest differs')
        view=torch.load(path,map_location='cpu',weights_only=True);old_view=torch.load(old_path,map_location='cpu',weights_only=True)
        for key in ['masks','semantic_descriptors','local_pixels','intrinsic','world_to_camera']:
            if not torch.equal(view[key],old_view[key]):raise ValueError('query/evaluation data differ: '+key)
        camera=SimpleNamespace(intrinsic=view['intrinsic'],world_to_camera=view['world_to_camera'],
            height=view['original_hw'][0],width=view['original_hw'][1])
        for index,mask in enumerate(view['masks']):
            row=dict(frame=entry['frame'],region=index,ious={condition:0. for condition in config['native_conditions']})
            inside=torch.where(erode(mask).flatten())[0]
            if not len(inside):inside=torch.where(mask.flatten())[0]
            if len(inside):
                positions=torch.linspace(0,len(inside)-1,min(16,len(inside))).round().long()
                packet=QueryPacket('image',semantic=view['semantic_descriptors'][index].to(device),
                                   local=view['local_pixels'][inside[positions]].to(device))
                for condition in config['native_conditions']:
                    later.semantic,later.local=channels[condition]
                    result=later.query(packet)
                    alpha=selected_alpha(later,result.selected,camera,height,width,manifest['world_units_per_metre'])
                    row['ious'][condition]=mask_iou(alpha>.5,mask.flatten().to(device))
            rows.append(row)
        print('native intervention',entry['frame'],len(view['masks']),flush=True)
    unchanged=all(torch.equal(value,later.state_dict()[key]) for key,value in protected.items())
    if not unchanged:raise RuntimeError('native intervention modified protected member state')
    report=dict(schema='radio_gs.query_field_native_transfer_report.v1',config=config,configuration_sha256=sha256(config_path),
        earlier_field_sha256=old_export['files']['field.pt']['sha256'],later_field_sha256=new_export['files']['field.pt']['sha256'],
        member_geometry_support_unchanged=unchanged,common_observed_elements=int(common.sum()),
        exact_query_input_parity=True,cases=len(rows),rows=rows,
        summary={condition:sum(r['ious'][condition] for r in rows)/len(rows) for condition in config['native_conditions']},
        benchmark_result=False,independent_final_test=False,promotion=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='rows'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
    print(json.dumps(run(a.config,a.output,a.device),indent=2))


if __name__=='__main__':main()
