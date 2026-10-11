"""Standalone unsupervised seed entry must replay every frozen seed-arm set."""
import builtins
import torch
from radio_gs.query_field.seed_scope import SeedScopeDecoder,query_text_seed_scope
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder
from scripts.research.run_query_field_region_scope import ROOT,read,write,receipt


def main():
    reports=[]
    for j in read(ROOT/'prediction_config.json')['jobs']:
        field,_=load_field(j['field']);decoder=SeedScopeDecoder(field);exclusion=JointExclusionDecoder(field);native=NativeTextDevice(field,'cuda')
        data=torch.load(ROOT/'predictions'/j['name']/'predictions.pt',weights_only=True)
        def denied(*a,**k):raise AssertionError('forbidden source or quality reader')
        field.region_carrier.quality.forward=denied;field.region_carrier.decoder=denied
        kwargs=dict(seed_decoder=decoder,exclusion_decoder=exclusion,native_reader=native)
        query_text_seed_scope(field,data['native_query_vectors'][0],data['generic_negative_vectors'],**kwargs)
        old=builtins.open
        try:
            builtins.open=denied
            for qi in range(len(data['texts'])):
                r=query_text_seed_scope(field,data['native_query_vectors'][qi],data['generic_negative_vectors'],**kwargs)
                assert torch.equal(r.selected,data['selected_by_control']['seed_addition'][:,qi])
                assert not r.diagnostics['additional_3d_label_model_used']
        finally:builtins.open=old
        reports.append(dict(scene=j['scene'],queries=len(data['texts']),all_selected_exact=True,
            python_open_denied=True,quality_model_and_pair_decoder_denied=True,os_sandbox=False))
        print(reports[-1],flush=True)
        del kwargs,r,field,decoder,exclusion,native,data;torch.cuda.empty_cache()
    report=dict(passed=True,scenes=reports,queries=sum(x['queries'] for x in reports),
        post_hoc_entry_extraction=True,no_new_predictions_or_metrics=True,default_promoted=False,
        note='Existing comparison fields still contain unused quality buffers; the seed entry does not read/execute them. First query warms Numba before open guard.',
        code=receipt('radio_gs/query_field/seed_scope.py'),validator=receipt(__file__))
    write(ROOT/'seed_entry_validation.json',report);write('docs/experiments/2026-10-11-seed-entry-validation.json',report)

if __name__=='__main__':torch.set_num_threads(4);main()
