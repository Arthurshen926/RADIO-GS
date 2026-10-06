"""Post-gate same-artifact diagnosis only; no new fit, query, mask or model."""
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from radio_gs.training.source_region_readback import region_render_weights
from radio_gs.utils.immutable_artifacts import sha256_file

OUT=Path('output/small_region_restoration_diagnostic_next32_v2')
PARENT=Path('output/small_region_restoration_next32')


def errors(pred,truth):
    delta=pred-truth;bias=delta.mean(1)
    return dict(mse=float(delta.square().mean()),centered_mse=float((delta-bias[:,None]).square().mean()),
        mean_bias=float(bias.mean()),mean_squared_row_bias=float(bias.square().mean()),
        teacher_top1_agreement=float((pred.argmax(1)==truth.argmax(1)).float().mean()),
        average_response=float(pred.mean()),teacher_average_response=float(truth.mean()))


@torch.inference_mode()
def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=False)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and any(v in str(args[0]) for v in (
            '.labels.ply','/lerf_ovs/label/','/eval_gt/','category_evaluation','ranking_evaluation','/official_annotations/')):
            raise RuntimeError('Benchmark data forbidden')
    sys.addaudithook(guard)
    reg=json.loads((PARENT/'registration.json').read_text());row=next(r for r in reg['cohort'] if r['key']=='lerf_figurines')
    report=json.loads((PARENT/row['key']/'report.json').read_text());root=Path(row['root']);bindings={}
    def load(path,digest):
        assert sha256_file(path)==digest,path;bindings[str(path)]=digest
        return torch.load(path,map_location='cpu',weights_only=False)
    info=json.loads((root/'teachers.json').read_text());auth=json.loads((root/'responsibility.json').read_text())
    teacher=load(root/info['teacher_files']['siglip']['path'],info['teacher_files']['siglip']['sha256'])
    ep=load(root/'episodes.pt',json.loads((root/'episodes.json').read_text())['sha256'])
    small=load(PARENT/row['key']/'small_teacher.pt',report['small_teacher_sha256'])
    new=load(PARENT/row['key']/'map.pt',report['map_sha256'])
    prior=Path('output/native_residual_basis_next30_v2')/row['key'];pr=json.loads((prior/'map/report.json').read_text())
    old=load(prior/'map/map.pt',pr['map_sha256']);basis=old['basis'][:64].float()
    observed=old['observed'];n=len(observed);native=F.normalize(teacher['descriptors'].float(),dim=-1)
    snative=F.normalize(small['descriptors'].float(),dim=-1)
    train=torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in ep['proposal_views']]);training=native[train]
    queries=dict(text=F.normalize(teacher['text_embeddings'].float(),dim=-1),
        image=training[torch.linspace(0,len(training)-1,min(512,len(training))).long()])
    fields={'baseline':F.normalize(old['features'][:,:64].float(),dim=-1),'candidate':F.normalize(new['features'].float(),dim=-1)}
    pools={stage:{mode:[] for mode in queries} for stage in ('projected_teacher','baseline','candidate','parent_teacher_mixture')}
    truths={mode:[] for mode in queries};rows=[]
    for frame in info['validation_frames']:
        vi=info['frames'].index(frame);rec=auth['views'][vi];hit=load(root/rec['relative_path'],rec['sha256'])
        pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();w=hit['base_weights'].float()
        ids=torch.where(ep['proposal_views']==vi)[0];sid=torch.where(small['proposal_views']==vi)[0]
        masks=ep['masks'][ids].flatten(1);smasks=small['masks'][sid].flatten(1)
        weights,supported,valid=region_render_weights(smasks,pid,gid,w,observed)
        parent_weights,_,_=region_render_weights(masks,pid,gid,w,observed)
        projections=snative[sid]@basis.T;direct=F.normalize(projections,dim=-1)
        predictions={stage:{mode:values@ (q@basis.T).T for mode,q in queries.items()}
            for stage,values in [('projected_teacher',direct),*[(arm,weights@f) for arm,f in fields.items()]]}
        raw={mode:snative[sid]@q.T for mode,q in queries.items()}
        for mode in queries:
            truths[mode].append(raw[mode][supported])
            for stage in predictions:pools[stage][mode].append(predictions[stage][mode][supported])
        for local in torch.where(supported)[0].tolist():
            smask=smasks[local];area=int(smask.sum());intersection=(masks&smask[None]).sum(1);parea=masks.sum(1)
            order=sorted(range(len(ids)),key=lambda j:(-int(intersection[j]),int(parea[j]),int(ids[j])))
            nearest=order[0];count=masks.sum(0);domain=smask&valid&(count>0)
            mixweight=(masks[:,domain].float()/count[domain][None]).mean(1) if domain.any() else torch.zeros(len(ids))
            mixture=F.normalize(mixweight@native[ids],dim=0)
            parent_scores={mode:mixture[None]@q.T for mode,q in queries.items()}
            for mode in queries:pools['parent_teacher_mixture'][mode].append(parent_scores[mode])
            sfeature=snative[sid[local]]
            item=dict(small_index=int(sid[local]),**small['source_rows'][int(sid[local])],coarse_pixels=area,
                parent_context_available=bool(domain.any()),overlap_parent_available=bool(intersection.max()>0),
                valid_pixel_fraction=float(valid[smask].float().mean()),parent_union_overlap=float((count[smask]>0).float().mean()),
                renderer_nonzero_elements=int((weights[local]>0).sum()),projected_teacher_norm=float(projections[local].norm()),
                parent_mixture_cosine=float(sfeature@mixture),tightest_max_overlap_parent_id=int(ids[nearest]),
                parent_overlap_fraction=float(intersection[nearest]/max(area,1)),parent_area_ratio=float(parea[nearest]/max(area,1)),
                parent_teacher_cosine=float(sfeature@native[ids[nearest]]),
                large_containing_parent_count=int(((intersection>=.9*area)&(parea>=2*area)).sum()),
                small_renderer_mass_in_parent_support=float(weights[local][parent_weights[nearest]>0].sum()),
                normalized_footprint_overlap=float(torch.minimum(weights[local],parent_weights[nearest]).sum()),
                raw_teacher_text_top1=teacher['queries'][int(raw['text'][local].argmax())],
                teacher_text_top1_after_projection=teacher['queries'][int(predictions['projected_teacher']['text'][local].argmax())],metrics={})
            for mode in queries:
                truth=raw[mode][local:local+1]
                item['metrics'][mode]={stage:errors(pred[mode][local:local+1],truth) for stage,pred in predictions.items()}
                item['metrics'][mode]['parent_teacher_mixture']=errors(parent_scores[mode],truth)
            rows.append(item)
    joined_truth={mode:torch.cat(v) for mode,v in truths.items()}
    joined={stage:{mode:torch.cat(v) for mode,v in modes.items()} for stage,modes in pools.items()}
    context_available=torch.tensor([r['parent_context_available'] for r in rows])
    metrics={stage:{mode:errors(p[context_available],joined_truth[mode][context_available]) if stage=='parent_teacher_mixture' else errors(p,joined_truth[mode]) for mode,p in modes.items()} for stage,modes in joined.items()}
    extra={arm:{mode:errors(joined[arm][mode],joined['projected_teacher'][mode]) for mode in queries} for arm in ('baseline','candidate')}
    for arm in ('baseline','candidate'):
        for mode in queries:
            assert abs(metrics[arm][mode]['mse']-report['metrics'][arm]['small'][mode]['mse'])<2e-7
    group={}
    for label,worse in [('image_regressions',True),('image_nonregressions',False)]:
        ids=[i for i,r in enumerate(rows) if (r['metrics']['image']['candidate']['mse']>r['metrics']['image']['baseline']['mse'])==worse]
        subset=[rows[i] for i in ids]
        group[label]=dict(count=len(ids),small_indices=[r['small_index'] for r in subset],
            means={key:sum(r[key] for r in subset)/max(len(subset),1) for key in ('valid_pixel_fraction','parent_union_overlap','projected_teacher_norm')},
            overlapping_parent_count=sum(r['overlap_parent_available'] for r in subset),
            valid_parent_context_count=sum(r['parent_context_available'] for r in subset),
            valid_parent_context_means={key:sum(r[key] for r in subset if r['parent_context_available'])/max(sum(r['parent_context_available'] for r in subset),1) for key in ('parent_mixture_cosine',)},
            overlapping_parent_means={key:sum(r[key] for r in subset if r['overlap_parent_available'])/max(sum(r['overlap_parent_available'] for r in subset),1) for key in ('parent_teacher_cosine','parent_area_ratio','small_renderer_mass_in_parent_support','normalized_footprint_overlap')},
            has_large_containing_parent=sum(r['large_containing_parent_count']>0 for r in subset),
            stage_metrics={stage:errors(modes['image'][ids],joined_truth['image'][ids]) for stage,modes in joined.items() if stage!='parent_teacher_mixture'} if ids else {})
    positive_delta=[max(r['metrics']['image']['candidate']['mse']-r['metrics']['image']['baseline']['mse'],0) for r in rows]
    total_increase=sum(positive_delta)
    regression_mass={label:sum(d for r,d in zip(rows,positive_delta) if predicate(r))/max(total_increase,1e-12) for label,predicate in [('has_large_containing_parent',lambda r:r['large_containing_parent_count']>0),('no_parent_mask_overlap',lambda r:not r['overlap_parent_available']),('any_parent_mask_overlap',lambda r:r['overlap_parent_available'])]}
    for item in rows:
        if not item['parent_context_available']:
            item['parent_mixture_cosine']=None
            for mode in queries:item['metrics'][mode]['parent_teacher_mixture']=None
        if not item['overlap_parent_available']:
            for field in ('tightest_max_overlap_parent_id','parent_area_ratio','parent_teacher_cosine','small_renderer_mass_in_parent_support','normalized_footprint_overlap'):item[field]=None
    result=dict(source_only=True,post_gate_diagnostic=True,no_new_training_or_model=True,benchmark_data_read=False,
        map_intervention_or_gate_changed=False,heldout_supported_regions=len(rows),all_heldout_regions=report['validation_small_regions'],
        parent_teacher_mixture_metric_population=int(context_available.sum()),image_positive_error_increment_shares=regression_mass,
        corrected_diagnostic='v1 preserved; absent parent context must be unknown, not zero teacher vector; no map or gate change',
        metrics=metrics,readback_deviation_from_direct_projected_teacher=extra,groups=group,rows=rows,
        metric_note='Stage errors share same native teacher target; not additive causal error. Parent mixture and containment are pseudo-evidence, not physical same-object labels. No object recognition GT is available.',
        diagnostic_definitions='Tightest max-overlap parent sorts intersection descending then area ascending; large containing parent means>=90%small-pixel overlap and>=2xarea. These summaries do not choose predictions.',
        inputs=bindings,implementation_sha256=sha256_file(__file__))
    (OUT/'figurines.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'metrics':metrics,'groups':group},indent=2))


if __name__=='__main__':main()
