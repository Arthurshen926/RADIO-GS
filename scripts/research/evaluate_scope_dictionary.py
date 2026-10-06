"""Evaluate dictionary retrieval using the existing benchmark scorers."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.models.native_scope_dictionary import NativeScopeDictionary,attach_scope_text_readout
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry


def load(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=False)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--element-results');p.add_argument('--key',required=True);p.add_argument('--base',default='output/scope_dictionary_next33');p.add_argument('--trained',action='store_true');p.add_argument('--pipeline',action='store_true');p.add_argument('--calibrated',action='store_true');p.add_argument('--readout',choices=['mixture','cosine_routed','anchored'],default='mixture');a=p.parse_args()
    torch.set_num_threads(4);out=Path(a.base)/a.key
    if a.trained:out=out/'trained'
    payload=load(out/'field.pt')
    if 'next42_parent' in payload:
        addition=payload;payload=load(addition['next42_parent']);payload['query_region_evidence']=addition['query_region_evidence']
    root=Path(payload['root']);scene=a.key.split('_',1)[1]
    field=NativeScopeDictionary(payload);parent=load(payload['base_field']);state=parent['state_dict']
    gr=json.loads(Path(payload['geometry_report']).read_text());g=gr['results']['independent_scale']
    geo=FrozenGaussianGeometry.load(g['geometry'],expected_sha256=g['geometry_sha256'],device='cuda')
    if a.key.startswith('scannet_'):
        from radio_gs.scannet_constants import NYU40_ID_TO_NAME,OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
        directory=root.parent.parent/('evaluation' if 'expanded_budget' in str(root) else 'evaluation_complete_next28')
        query=load(directory/'queries.pt');q=F.normalize(query['encoding'].float().cuda(),dim=1)
        classes=sorted(set(sum([SPLITS[s] for s in ('19','15','10')],[])))
        texts=query['texts']
    else:
        directory=root.parent.parent/'evaluation';query=load(directory/'text_online.encodings.pt')[0]
        q=F.normalize(query['encoding'].half().float().cuda(),dim=1);texts=query['texts']
    native=F.normalize(state['latent'][:,:64].float().cuda(),dim=1)@(q@state['native_decoder.weight'].float().cuda()).T
    scores=dict(native64=native,dictionary=field.scores(q,pool=a.readout,anchor=(state['latent'][:,:64].float().cuda(),state['native_decoder.weight'].float().cuda())))
    if a.readout=='mixture' and field.scale_prior is not None:scores['routed_dictionary']=field.scores(q,pool='routed')
    if a.readout!='mixture':
        out=out/a.readout;out.mkdir(exist_ok=True)
    if a.pipeline:
        from radio_gs.models.element_retrieval_field import ElementRetrievalField
        model=ElementRetrievalField(**parent['architecture']).cuda();model.load_state_dict(state);model.eval()
        if a.key.startswith('scannet_'):
            from radio_gs.querying.source_accepted_category import source_accepted_category_logits
            scope=json.loads((root/'bounded_fallback_withheld/report.json').read_text())
            admission=json.loads((Path(payload['base_field']).parent/'report.json').read_text())
            def predict():
                calibrated=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q))
                return source_accepted_category_logits(calibrated,lambda:model.logits(q,'text'),scope_receipt=scope,admission_receipt=admission,max_correction=1.)[0]
        else:
            def predict():return model.retrieve(q,'text')['scores']
        baseline=predict()
        if 'region_native_features' in payload:
            def native_prediction():
                value=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q))
                return value if a.key.startswith('scannet_') else value.sigmoid()*model.observed[:,None]
            native_baseline=native_prediction()
        if 'scope_state' in payload:model.grounded_transfer.load_state_dict(payload['scope_state'])
        if 'dense_surface_field' in payload:
            from radio_gs.querying.dense_surface_field import attach_dense_surface_field
            attach_dense_surface_field(model,payload)
        elif 'source_kernel_extent' in payload:
            from radio_gs.querying.source_kernel_extent import attach_source_kernel_extent
            attach_source_kernel_extent(model,payload)
        elif 'region_native_features' in payload:
            from radio_gs.querying.region_native_field import attach_region_native_field
            attach_region_native_field(model,payload)
        elif 'native_scope_union' in payload:
            from radio_gs.querying.native_scope_union import attach_native_scope_union
            attach_native_scope_union(model,payload)
        elif 'scope_hypergraph' in payload:
            from radio_gs.querying.scope_hypergraph import attach_scope_hypergraph
            attach_scope_hypergraph(model,payload)
        else:attach_scope_text_readout(model,field)
        if a.calibrated:
            calibration=load(Path(a.base)/a.key/'calibration.pt')
            original=model.native_query_thresholds['text'].forward
            def threshold(query):
                delta=F.normalize(query,dim=-1)@calibration['weight'].cuda()+calibration['bias'].cuda()
                return original(query)+delta.clamp(-calibration['max_delta'],calibration['max_delta'])
            model.native_query_thresholds['text'].forward=threshold
        scores=dict(retained_pipeline=baseline,hierarchical_pipeline=predict())
        if ('source_kernel_extent' in payload or 'dense_surface_field' in payload) and a.key.startswith('scannet_'):
            scores['hierarchical_pipeline']=model.logits(q,'text')
        if 'region_native_features' in payload:
            scores.update(retained_native=native_baseline,reconstructed_native=native_prediction())
        if a.key.startswith('scannet_') and 'category_competition' in payload:
            from radio_gs.querying.category_competition import correct_category_logits
            scores['hierarchical_pipeline']=correct_category_logits(scores['hierarchical_pipeline'],q,payload['category_competition'])
        if 'query_region_evidence' in payload:
            from radio_gs.querying.query_region_evidence import load_runtime,corrected_scores
            addon=payload['query_region_evidence']
            # Bind every arm to the SAME immutable next38 prior. GPU sparse
            # accumulation can differ at the last bit on a fresh parent replay.
            frozen_parent=load(addon['parent_scores'])
            assert frozen_parent['texts']==texts
            replay={}
            for name,value in scores.items():
                exact=frozen_parent['scores'][name].to(value.device)
                error=float((value-exact).abs().max())
                replay[name]=dict(recomputed_exact=torch.equal(value,exact),max_abs_recompute_error=error)
                assert torch.allclose(value,exact,atol=2e-6,rtol=0), 'parent recipe/calibration mismatch'
                scores[name]=exact
            (out/'next42_parent_replay.json').write_text(json.dumps(dict(frozen_scores=addon['parent_scores'],scores=replay,frozen_prior_exact=True),indent=2))
            if addon['arm']!='B0':
                runtime=load_runtime(addon,q.device)
                categorical=a.key.startswith('scannet_')
                import time
                parent_scores=scores['hierarchical_pipeline']
                base_logits=None if categorical else torch.where((parent_scores>0)&(parent_scores<1),torch.logit(parent_scores),model.logits(q,'text'))
                saved={k:v.clone() for k,v in runtime.head.state_dict().items()}
                runtime.head.net[-1].weight.zero_();runtime.head.net[-1].bias.zero_()
                zero=corrected_scores(runtime,q,parent_scores,categorical,base_logits=base_logits)
                assert torch.equal(zero,parent_scores), 'next42 zero correction must be exact'
                runtime.head.load_state_dict(saved)
                torch.cuda.synchronize();started=time.monotonic()
                scores['hierarchical_pipeline']=corrected_scores(runtime,q,parent_scores,categorical,base_logits=base_logits)
                torch.cuda.synchronize();elapsed=time.monotonic()-started
                (out/'next42_runtime.json').write_text(json.dumps(dict(zero_correction_exact=True,queries=len(q),seconds=elapsed,seconds_per_query=elapsed/len(q),max_abs_score_change=float((scores['hierarchical_pipeline']-parent_scores).abs().max()),changed_fraction=float((scores['hierarchical_pipeline']!=parent_scores).float().mean())),indent=2))
        out=out/('pipeline_calibrated' if a.calibrated else 'pipeline');out.mkdir(exist_ok=True)
    if a.element_results:
        online=load(a.element_results)
        if online['texts']!=texts:raise ValueError('runtime query order differs from evaluation')
        scores['hierarchical_pipeline']=online['scores'].to(q.device)
    selection_scores=dict(scores)
    if a.element_results and 'selection_scores' in online:
        selection_scores['hierarchical_pipeline']=online['selection_scores'].to(q.device)
    selection_geo=geo
    selection_observed=field.observed
    semantic_geo=geo
    semantic_observed=field.observed
    if a.element_results and 'semantic_geometry' in online:
        binding=online['semantic_geometry']
        semantic_geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
        semantic_observed=online['semantic_observed'].to(q.device)
    if a.element_results and 'selection_geometry' in online:
        binding=online['selection_geometry']
        selection_geo=FrozenGaussianGeometry.load(binding['path'],expected_sha256=binding['sha256'],device='cuda')
        selection_observed=online['selection_observed'].to(q.device)
    saved_scores=dict(scores={k:v.cpu() for k,v in scores.items()},texts=texts)
    if a.element_results and 'semantic_geometry' in online:
        saved_scores['semantic_geometry']=online['semantic_geometry']
        saved_scores['semantic_observed']=online['semantic_observed']
    if a.element_results and 'selection_scores' in online:saved_scores['selection_scores']=online['selection_scores']
    if a.element_results and 'selection_geometry' in online:
        saved_scores['selection_geometry']=online['selection_geometry']
        saved_scores['selection_observed']=online['selection_observed']
    torch.save(saved_scores,out/'scores.pt')
    if a.key.startswith('scannet_'):
        from radio_gs.scripts.eval_scannet_pointcloud_radio_gs import _read_label_ply
        from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import assign_vala_pseudo_labels,volume_weighted_split_metrics
        predictions={}
        for name,values in scores.items():
            predictions[name]={}
            for split in ('19','15','10'):
                ids=SPLITS[split];labels=torch.tensor(ids,device='cuda')[values[:,[classes.index(c) for c in ids]].argmax(1)]
                predictions[name][split]=torch.where(field.observed,labels,0).cpu().numpy()
            np.savez_compressed(out/(name+'.npz'),**predictions[name])
        xyz,labels=_read_label_ply(Path('/mnt/pool/sqy/3d_understanding/scannet_og')/scene/(scene+'_vh_clean_2.labels.ply'))
        pseudo,_=assign_vala_pseudo_labels(geo.xyz.cpu().numpy(),geo.scales.cpu().numpy(),geo.quaternions.cpu().numpy(),xyz,labels,radius_factor=5.,candidate_k=1000,fallback_k=1,class_balance=True,chunk_size=512)
        weights=(geo.scales.prod(1)*geo.opacities.flatten()).cpu().numpy()
        metrics={name:{s:volume_weighted_split_metrics(pseudo,pred[s],weights,SPLITS[s]) for s in ('19','15','10')} for name,pred in predictions.items()}
    else:
        from radio_gs.evaluation.lerf_readouts import evaluate_lerf_readouts
        metrics=evaluate_lerf_readouts(scene=scene,out=out,texts=texts,scores=scores,
            selection_scores=selection_scores,geo=geo,default_observed=field.observed,
            semantic_geo=semantic_geo,semantic_observed=semantic_observed,
            selection_geo=selection_geo,selection_observed=selection_observed,
            camera_report=gr['source_report'],camera_report_sha256=gr['source_report_sha256'],pipeline=a.pipeline)
    (out/'evaluation.json').write_text(json.dumps(dict(key=a.key,metrics=metrics,development=True,separate_selection_readout=bool(a.element_results and 'selection_scores' in online)),indent=2)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
