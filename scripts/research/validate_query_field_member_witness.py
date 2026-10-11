"""Cold first-scene public API parity with source file reads denied."""
import builtins
from unittest.mock import patch
import numpy as np
from scipy import sparse
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.member_energy import solve_member_energy,query_text_member_energy
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder
from scripts.research.run_query_field_member_witness import ROOT,read,write


def main():
    read(ROOT/'predictions_complete.json');j=read(ROOT/'prediction_config.json')['jobs'][0]
    field,_=load_field(j['field'],'cuda');decoder=JointExclusionDecoder(field)
    data=torch.load(ROOT/'predictions'/j['name']/'predictions.pt',weights_only=True)
    solve_member_energy(sparse.csc_matrix([[2.,2.]]),[.9,.1],np.array([True]),np.array([False]),
        [1.],[10.,1.],cutoff=1.,engine='compiled',addition_allowed=np.ones(2,bool))
    def denied(*args,**kwargs):raise AssertionError('query attempted file read')
    with patch.object(builtins,'open',denied):
        for qi in range(len(data['texts'])):
            r=query_text_member_energy(field,data['native_query_vectors'][qi].cuda(),data['generic_negative_vectors'].cuda(),
                decoder=decoder,addition_policy='positive_witness',engine='compiled')
            assert torch.equal(r.selected.cpu(),data['selected_by_control']['witnessed_energy'][:,qi]),data['texts'][qi]
    report=dict(passed=True,scene=j['scene'],queries=len(data['texts']),selected_exact=True,
        source_reads_denied=True,synthetic_kernel_warmup=True,operating_system_sandbox=False)
    write(ROOT/'cold_validation.json',report);print(report,flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60);main()
