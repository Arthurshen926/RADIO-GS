"""Cold source-free replay of diagnostic text scope cache (no teacher/GT reads)."""
import json,sys
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.source_scope_consensus import refine_source_scope_consensus
from radio_gs.utils.immutable_artifacts import sha256_file

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0])
        if any(s in path for s in ['caption_grounded_regions.pt','caption_region_evidence.pt','source_grounded_text.pt','episodes.pt','/lerf_ovs/label/','/eval_gt/']):
            raise RuntimeError('teacher/target input forbidden in cold scope readout: '+path)
sys.addaudithook(guard)
torch.set_num_threads(4);rows=[]
with torch.no_grad():
    for scene in ['figurines','ramen','teatime','waldo_kitchen']:
        root=Path('output/caption_adjoint_allsource_next20')/scene;base=Path('output/full_field_rebuild_next19')/scene
        sr=json.loads((root/'report.json').read_text());assert sha256_file(root/'uplift.pt')==sr['uplift_sha256']
        up=torch.load(root/'uplift.pt',map_location='cpu',weights_only=False);original=torch.load(base/'evaluation/scores.pt',map_location='cpu',weights_only=False);expected=torch.load(root/'evaluation/scores.pt',map_location='cpu',weights_only=False)
        fp=base/'reconstruction/field/bounded_fallback_admission/field.pt';assert sha256_file(fp)==sr['field_sha256'];payload=torch.load(fp,map_location='cpu',weights_only=False)
        field=ElementRetrievalField(**payload['architecture']);field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
        q=torch.load(base/'evaluation/text_online.encodings.pt',map_location='cpu',weights_only=False)[0]['encoding'].half().float()
        index,concept,_=field.grounded_transfer.route(q);accepted=concept&((original['scores']['final']-original['scores']['native_compact_calibrated']).abs().amax(0)>2e-5)
        result={n:original['scores']['final'].clone() for n in ['final','lifted_calibrated','local_calibrated']}
        result['local_calibrated']=refine_source_scope_consensus(result['final'],original['observed'],up['mean'],up['variance'],up['count'],index,accepted,modality='text',enabled=True)
        for qi in torch.where(accepted)[0].tolist():
            j=int(index[qi]);mean=up['mean'][:,j];count=up['count'][:,j];base_scores=result['final'][:,qi]
            result['lifted_calibrated'][:,qi]=torch.where(count>0,mean,base_scores)*original['observed']
            reliable=original['observed']&(count>=2)&(up['variance'][:,j]<=.04)&((mean>=.75)|(mean<=.25))
            assert torch.equal(result['local_calibrated'][~reliable,qi],base_scores[~reliable])
        for arm in result:assert torch.equal(result[arm],expected['scores'][arm]),(scene,arm)
        rows.append(dict(scene=scene,exact=True,query_count=len(q),routed_count=int(accepted.sum()),uplift_bytes=(root/'uplift.pt').stat().st_size,field_sha256=sr['field_sha256'],scores_sha256=sha256_file(root/'evaluation/scores.pt')))
        print(scene,'cold exact',flush=True)
        del field,payload,original,expected,result,up
p=Path('output/caption_adjoint_allsource_next20/cold_runtime_module.json');assert not p.exists();p.write_text(json.dumps(dict(complete=True,source_teacher_target_reads_forbidden=True,runtime_module_sha256=sha256_file("radio_gs/querying/source_scope_consensus.py"),rows=rows,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
