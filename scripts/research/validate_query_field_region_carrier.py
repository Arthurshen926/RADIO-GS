"""Cold field state preservation and file-denied public query replay."""
import argparse,builtins,gc
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.region_carrier import query_text_region_scope
from radio_gs.query_field.joint_exclusion import JointExclusionDecoder
from scripts.research.run_query_field_region_scope import ROOT,FIELDS,PARENT,ARMS,read,write,receipt


def run(index):
    cfg=read(ROOT/'prediction_config.json');j=cfg['jobs'][index]
    field,_=load_field(j['field'])
    previous=read(PARENT/'prediction_config.json')['jobs'][index]
    old=torch.load(Path(previous['field'])/'field.pt',weights_only=True)
    new=field.state_dict()
    assert set(new)-set(old)=={'region_carrier.'+k for k in field.region_carrier.state_dict()}
    for k,v in old.items():assert torch.equal(v,new[k]),k
    count=len(old);del old;gc.collect()
    report=dict(scene=j['scene'],cold_parent_tensors_exact=True,parent_tensor_count=count,new_state_self_contained=True,
                manifest=receipt(Path(j['field'])/'manifest.json'),validator=receipt(__file__))
    b=field.region_carrier.matrix('membership');v=field.region_carrier.matrix('visibility')
    graph=(b.T@b).tocsr();graph.setdiag(0);graph.eliminate_zeros()
    frames=field.region_carrier.frames.numpy();coo=graph.tocoo()
    cross=coo.row[frames[coo.row]!=frames[coo.col]]
    nonempty=np.asarray(b.sum(0)).ravel()>0
    visible=np.asarray(v.sum(1)).ravel();member=np.asarray(b.sum(1)).ravel()>0
    components,labels=connected_components(graph,directed=False)
    report['graph_diagnostics']=dict(region_count=b.shape[1],nonempty_regions=int(nonempty.sum()),
        regions_with_cross_view_overlap=len(np.unique(cross)),directed_overlap_edges=graph.nnz,
        directed_cross_view_edges=len(cross),components=components,
        largest_component_regions=int(np.bincount(labels).max()),
        observed_rows=int(field.observed.sum()),any_reliable_view_rows=int((visible>0).sum()),
        at_least_two_reliable_views_rows=int((visible>=2).sum()),any_region_members=int(member.sum()),
        total_gaussians=len(visible))
    del b,v,graph
    if index==0:
        data=torch.load(ROOT/'predictions'/j['name']/'predictions.pt',weights_only=True)
        decoder=field.region_carrier.decoder(field.xyz);exclusion=JointExclusionDecoder(field);native_reader=NativeTextDevice(field,'cuda')
        # Warm Numba's fixed-exclusion kernel before denying file opens. There
        # is no answer cache; all following queries execute the public API.
        query_text_region_scope(field,data['native_query_vectors'][0],data['generic_negative_vectors'],region_decoder=decoder,exclusion_decoder=exclusion,native_reader=native_reader)
        original=builtins.open
        def denied(*a,**k):raise AssertionError('runtime source read')
        try:
            builtins.open=denied
            for qi in range(len(data['texts'])):
                values,_=query_text_region_scope(field,data['native_query_vectors'][qi],data['generic_negative_vectors'],region_decoder=decoder,exclusion_decoder=exclusion,native_reader=native_reader)
                for a in ARMS:assert torch.equal(values[a]>.5,data['selected_by_control'][a][:,qi])
        finally:builtins.open=original
        report.update(file_denied_queries=len(data['texts']),arms=len(ARMS),all_sets_exact=True,os_sandbox=False,warmup='first actual query before Python open guard')
    write(FIELDS/j['scene']/'cold_validation.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
