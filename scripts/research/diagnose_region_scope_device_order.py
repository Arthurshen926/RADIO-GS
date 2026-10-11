"""Explain a failed control replay before any new benchmark scoring."""
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.scope_confidence import native_text_evidence,native_scope_confidence
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder
from scripts.research.run_query_field_relations import read,write,receipt


def main():
    root=Path('output/query_field_20261011/binary_region_scope_v1');j=read(root/'prediction_config.json')['jobs'][1]
    f,_=load_field(j['field']);old=torch.load(Path('output/query_field_20261011/member_energy_v2/predictions')/j['name']/'predictions.pt',weights_only=True)
    qi=old['texts'].index('hand');q=old['native_query_vectors'][qi];neg=old['generic_negative_vectors']
    cpu,cp=native_text_evidence(f,q,neg);gpu,gp=NativeTextDevice(f,'cuda')(q,neg)
    ca,_=native_scope_confidence(f,cpu,cp);ga,_=native_scope_confidence(f,gpu,gp)
    decoder=JointExclusionDecoder(f);reference=old['selected_by_control']['frozen_visibility'][:,qi];rows={}
    for name,initial,accepted in [('cpu_cpu',cpu,ca),('cpu_gpu',cpu,ga),('gpu_cpu',gpu,ca),('gpu_gpu',gpu,ga),('stored_gpu',old['membership_by_control']['semantic_initial'][:,qi],ga)]:
        result,_=decoder.decode(initial,accepted,joint=False)
        rows[name]=dict(different_selected=int(((result>.5)!=reference).sum()))
    report=dict(scene=j['scene'],query='hand',gt_access=False,
        native_max_cpu_gpu_difference=float((cpu-gpu).abs().max()),native_initial_set_difference=int(((cpu>.5)!=(gpu>.5)).sum()),
        admitted_region_difference=int((ca!=ga).sum()),cases=rows,code=receipt(__file__))
    write(root/'device_order_diagnosis.json',report);print(report,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
