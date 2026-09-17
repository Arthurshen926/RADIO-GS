"""Source-only decision-boundary transfer; leave-one-concept-out ridge choice."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_threshold import fit_threshold_ridge
from radio_gs.training.retrieval_metrics import binary_ranking_metrics
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'scale_affinity_v1_direct_control';out=root/'native_query_threshold_control';out.mkdir(exist_ok=False)
    report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False)
    receipt=json.loads((root/'episodes.json').read_text());assert sha256_file(root/'episodes.pt')==receipt['sha256']
    data=torch.load(root/'episodes.pt',weights_only=False)
    assert sha256_file(root/'grounded_text_supervision.pt')==report['grounded_text_supervision_sha256']
    gt=torch.load(root/'grounded_text_supervision.pt',weights_only=False)
    assert gt['episode_sha256']==receipt['sha256'] and gt['geometry_binding']==payload['geometry_binding']
    field=ElementRetrievalField(**payload['architecture']).to(a.device)
    field.load_state_dict(payload['state_dict']);field.requires_grad_(False)
    q=data['queries']['text'].float().to(a.device)
    records=[];admitted=[];thresholds=[]
    with torch.no_grad():
        scores=field.native_cosine(q);scale=field.readout_log_scale['text'].clamp(-4.,6.).exp()
        for qi in sorted({t['text_index'] for t in gt['tasks']}):
            views=sorted({t['target_view'] for t in gt['tasks'] if t['text_index']==qi})
            pos=gt['positive_elements'][qi].to(a.device);neg=gt['negative_elements'][qi].to(a.device)
            known=(pos^neg)&field.observed&data['seen'][views].any(0).to(a.device)
            ps=scores[known&pos,qi];ns=scores[known&neg,qi]
            ranking=binary_ranking_metrics(scores[known,qi],pos[known])
            row=dict(query_index=qi,positive_count=len(ps),negative_count=len(ns),source_auroc=ranking['auroc'])
            if min(len(ps),len(ns))<20 or ranking['auroc']<=.5:
                records.append(dict(row,admitted=False));continue
            lo,hi=-1.,1.
            for _ in range(40):
                mid=(lo+hi)/2
                mass=((scale*(ps-mid)).sigmoid().mean()+(scale*(ns-mid)).sigmoid().mean())/2
                if mass>.5:lo=mid
                else:hi=mid
            threshold=(lo+hi)/2
            records.append(dict(row,admitted=True,threshold=threshold))
            admitted.append(qi);thresholds.append(threshold)
        features=q[admitted];targets=torch.tensor(thresholds,device=a.device)
        assert len(targets)>=4
        candidates=[.01,.1,1.,10.];validation=[]
        for penalty in candidates:
            errors=[]
            for i in range(len(targets)):
                keep=torch.arange(len(targets),device=a.device)!=i
                w,b=fit_threshold_ridge(features[keep],targets[keep],penalty)
                predicted=F.normalize(features[i],dim=-1)@w+b
                errors.append(float((predicted-targets[i]).square()))
            validation.append(sum(errors)/len(errors))
        chosen=min(range(len(candidates)),key=validation.__getitem__)
        w,b=fit_threshold_ridge(features,targets,candidates[chosen])
    architecture=dict(payload['architecture'],native_query_thresholds={'text':{'lower':float(targets.min()),'upper':float(targets.max())}})
    state=dict(payload['state_dict'])
    state.update({'native_query_thresholds.text.linear.weight':w.cpu()[None],
                  'native_query_thresholds.text.linear.bias':b.cpu()[None],
                  'native_query_thresholds.text.bounds':torch.tensor([float(targets.min()),float(targets.max())])})
    cold=ElementRetrievalField(**architecture);cold.load_state_dict(state,strict=True)
    torch.save(dict(payload,architecture=architecture,state_dict=state),out/'field.pt')
    record=dict(report,schema='native_query_threshold_control.v1',field_sha256=sha256_file(out/'field.pt'),
                parent_field_sha256=report['field_sha256'],source_only=True,benchmark_data_used=False,
                native_text_image_max_error=None,native_cosine_exactly_preserved=True,
                image_prompt_unchanged=True,all_parent_tensors_unchanged=True,
                source_query_fits=records,ridge_candidates=candidates,leave_one_concept_out_mse=validation,
                selected_ridge=candidates[chosen],implementation_sha256=sha256_file(__file__),
                readout_implementation_sha256=sha256_file('radio_gs/models/native_query_threshold.py'))
    (out/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(scene=a.scene,thresholds=thresholds,ridge=candidates[chosen],loo_mse=validation)),flush=True)


if __name__=='__main__':main()
