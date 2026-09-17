import json
import subprocess
import sys
from pathlib import Path
import torch
import pytest
from radio_gs.utils.immutable_artifacts import sha256_file

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/audits/rebind_source_grounding_inventory.py'


def fixture(tmp_path, change=None):
    parent=tmp_path/'old';root=tmp_path/'new';parent.mkdir();root.mkdir()
    rgb=tmp_path/'frame.jpg';rgb.write_bytes(b'original source bytes')
    source={'image_dir':str(tmp_path),'frames':[{'frame_idx':1,'source_file':'frame.jpg','source_sha256':sha256_file(rgb)}]}
    authority={'metadata':{'frozen_geometry_binding':{'id':'same'},'feature_height':2,'feature_width':2}}
    for r in (parent,root):
        (r/'source_frames.json').write_text(json.dumps(source))
        (r/'responsibility.json').write_text(json.dumps(authority))
        (r/'native_query_encoder.json').write_text('{}')
        torch.save({'queries':['a photo of a bowl'],'descriptors':torch.ones(1 if r==parent else 2,3)},r/'siglip.pt')
        info={'train_frames':[1],'authority_sha256':sha256_file(r/'responsibility.json'),'teacher_files':{'siglip':{'path':'siglip.pt','sha256':sha256_file(r/'siglip.pt')}}}
        (r/'teachers.json').write_text(json.dumps(info))
    g={'schema':'source_grounded_text.v1','teacher_manifest_sha256':sha256_file(parent/'teachers.json'),'source_only':True,'benchmark_labels_used':False,'frames':[1],'queries':['a photo of a bowl'],'grounding_phrases':['bowl'],'geometry_binding':{'id':'same'},'probability':torch.rand(1,1,2,2),'source_records':[{'frame':1,'sha256':sha256_file(rgb)}]}
    torch.save(g,parent/'source_grounded_text.pt');(parent/'source_grounded_text.json').write_text(json.dumps({'sha256':sha256_file(parent/'source_grounded_text.pt')}))
    if change=='rgb':rgb.write_bytes(b'changed image')
    if change=='encoder':(root/'native_query_encoder.json').write_text('{"different":true}')
    if change=='vocabulary':
        torch.save({'queries':['a photo of a egg']},root/'siglip.pt');info=json.loads((root/'teachers.json').read_text());info['teacher_files']['siglip']['sha256']=sha256_file(root/'siglip.pt');(root/'teachers.json').write_text(json.dumps(info))
    return parent,root,g


def test_rebind_only_region_inventory_and_preserve_predictions(tmp_path):
    parent,root,g=fixture(tmp_path)
    subprocess.run([sys.executable,str(SCRIPT),'--parent',str(parent),'--root',str(root)],check=True,capture_output=True)
    rebound=torch.load(root/'source_grounded_text.pt',weights_only=False)
    assert torch.equal(g['probability'],rebound['probability'])
    assert rebound['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
    assert rebound['inventory_rebinding']['source_prediction_sha256']==sha256_file(parent/'source_grounded_text.pt')


@pytest.mark.parametrize('change',['rgb','encoder','vocabulary'])
def test_changed_inference_inputs_reject_cache_rebinding(tmp_path,change):
    parent,root,_=fixture(tmp_path,change)
    result=subprocess.run([sys.executable,str(SCRIPT),'--parent',str(parent),'--root',str(root)],capture_output=True)
    assert result.returncode!=0 and not (root/'source_grounded_text.pt').exists()
