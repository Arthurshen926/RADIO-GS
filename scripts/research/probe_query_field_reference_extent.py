"""Cold first-case query with historical scene files and tensor loads denied."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import builtins
import io
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.runtime import QueryFieldRuntime
from radio_gs.query_field.geometry import projection
from radio_gs.query_field.adapters import scribble_packet
from radio_gs.query_field.public_nvos import load_strokes
from scripts.research.run_query_field_reference_extent import ROOT,design
from scripts.research.run_query_field_relations import read,write,receipt,check


def main():
    d=design();j=d['jobs'][0];cfg=read(j['configuration']['path'])
    job=next(x for x in cfg['jobs'] if x['name'].endswith('__shared_prior'))
    field,_=load_field(job['field'],'cuda');fixture=torch.load(job['camera_fixture'],weights_only=True)
    camera=SimpleNamespace(**fixture['reference']);h,w=cfg['projection_hw']
    strokes,_=load_strokes(job['positive'],job['negative'],(h,w))
    raw=np.asarray(Image.open(job['reference_mask']));fg=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
    reference=F.interpolate(torch.from_numpy(fg.copy()).float()[None,None],size=(h,w),mode='area')[0,0]
    geometry=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,
                             get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
    _,op,_,_=projection(geometry,camera,h,w,field.world_units_per_metre)
    packet,_=scribble_packet(op,reference,None,preserve_rays=field.support_aggregation=='expected_energy',
                             negative_rays=field.negative_rule=='competitive_rays')
    del fixture,raw,fg,strokes,op
    roots=[os.path.realpath('output'),'/mnt/pool/sqy/3d_understanding']
    def guarded(original):
        def call(file,*args,**kwargs):
            if isinstance(file,(str,bytes,os.PathLike)):
                path=os.path.realpath(os.fsdecode(file))
                if any(path==r or path.startswith(r+os.sep) for r in roots):raise AssertionError('scene file opened during query: '+path)
            return original(file,*args,**kwargs)
        return call
    with patch('builtins.open',guarded(builtins.open)),patch('io.open',guarded(io.open)),patch('torch.load',side_effect=AssertionError('tensor load during query')):
        result=QueryFieldRuntime(field).reference_extent(packet,camera,reference,initializer='prior',decision_cutoff=.5)
    # Comparisons only after the independently executed public query.
    p=ROOT/'predictions'/j['name']/job['name'];record=read(p/'prediction.json')
    check(dict(path=str(p/'predictions.pt'),sha256=record['predictions_sha256']))
    old=torch.load(p/'predictions.pt',weights_only=True)
    assert torch.equal(result.selected.cpu(),old['selected'])
    assert torch.equal(result.membership_scores.cpu(),old['membership'])
    report=dict(passed=True,case=j['name'],rule='fixed first case shared_prior',member_values_exact=True,
        selected_bits_exact=True,query_scene_file_roots_denied=roots,torch_load_denied=True,
        code_and_compiler_cache_io_allowed=True,os_sandbox=False,cold_process=True,
        actual_query_warmup=False,target_camera_passed_to_decoder=False,reference_rgb_used=False,
        prediction=receipt(p/'prediction.json'),code=receipt(__file__))
    write(ROOT/'cold_probe.json',report);write('docs/experiments/2026-10-12-reference-extent-cold-probe.json',report)
    print('PASS cold current-reference public query',flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
