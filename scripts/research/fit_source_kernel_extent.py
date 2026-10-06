"""Fit actual source masks through frozen rendering, then native-text kernels.

Each phrase's observed masks define a text-query union within that source view;
different phrases remain separate, including whole/part descriptions. Missing
detections are unknown. Negatives require another source mask with lower native
visual affinity. A reserved source-view subset never updates the extent field.
"""
import argparse
import json
import math

def clean_json(x):
    if isinstance(x,dict):return {k:clean_json(v) for k,v in x.items()}
    if isinstance(x,list):return [clean_json(v) for v in x]
    return None if isinstance(x,float) and not math.isfinite(x) else x

from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.source_kernel_extent import attach_kernel_native, native_probability
from radio_gs.training.partial_scope_metrics import extent_measures, aggregate, episode_rows, select_ridge, odds_probability


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def extent_training_states(base_logits, examples, held, denominator, visible_denominator,
                           steps, prior=.1, loss='robust', readout='odds',
                           prior_domain='known', consistency=None, carrier='elements',
                           membership=None, transpose=None, majorizer=None,
                           prior_majorizer=None, spatial_coefficients=None):
    """Yield pre-readout state; use deployed probabilities only for the forward.

    The adjoint update uses a straight-through identity approximation for the
    odds derivative, NOT the exact clamp/logit/sigmoid chain rule. The prior
    regularizes the same pre-readout correction that is eventually serialized.
    """
    baseline=base_logits.sigmoid()
    pre_state=baseline.clone()
    for step in range(steps):
        value=odds_probability(base_logits,pre_state-baseline) if readout=='odds' else pre_state
        gradient=torch.zeros_like(pre_state)
        for e in examples:
            if e['view'] in held:
                continue
            residual=e['target']-torch.sparse.mm(e['op'],value[:,e['columns']])
            if loss=='robust':residual=residual.clamp(-.25,.25)
            gradient[:,e['columns']] += torch.sparse.mm(e['adjoint'],residual*e['known'])
        if consistency is not None:gradient*=consistency
        if carrier=='scopes':
            gradient=torch.sparse.mm(transpose,gradient)-prior*prior_majorizer*spatial_coefficients
            spatial_coefficients.add_(.95*gradient/(majorizer+prior*prior_majorizer))
            correction=torch.sparse.mm(membership,spatial_coefficients)
            if consistency is not None:correction*=consistency
            pre_state=(baseline+correction).clamp(0,1)
        else:
            regularizer=visible_denominator if prior_domain=='visible' else denominator
            gradient -= prior*regularizer*(pre_state-baseline)
            pre_state.add_(.95*gradient/(denominator+prior*regularizer).clamp_min(1e-8)).clamp_(0,1)
        value=odds_probability(base_logits,pre_state-baseline) if readout=='odds' else pre_state
        yield step+1, pre_state, spatial_coefficients, value


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--key', required=True)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--output', default='output/source_kernel_extent_next40')
    p.add_argument('--prior', type=float, default=.1)
    p.add_argument('--loss', choices=['quadratic', 'robust'], default='robust')
    p.add_argument('--baseline', choices=['native', 'pipeline'], default='native')
    p.add_argument('--save-training', action='store_true')
    p.add_argument('--carrier', choices=['elements','scopes'], default='elements')
    p.add_argument('--evidence-policy', choices=['observations','consistent_views'], default='observations')
    p.add_argument('--readout', choices=['probability','odds'], default='odds')
    p.add_argument('--negative-policy', choices=['affinity','disjoint_regions'], default='affinity')
    p.add_argument('--interpolation', choices=['signed_kernel','conditional_evidence'], default='signed_kernel')
    p.add_argument('--prior-domain', choices=['known','visible'], default='known')
    p.add_argument('--selection-rule', choices=['legacy_visible_iou','partial_known_brier'], default='legacy_visible_iou')
    p.add_argument('--dual-selection', action='store_true')
    a = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    data = load(Path('output/taskwise_scope_composition_next38' if a.baseline=='pipeline' else 'output/scope_query_modes_next34')/a.key/'field.pt')
    competition=data.pop('category_competition',None)
    data['kernel_baseline']=a.baseline
    data['kernel_native_calibration'] = load(Path('output/scope_query_modes_next34')/a.key/'calibration.pt')
    root = Path(data['root'])
    if a.key.startswith('scannet_') and a.baseline=='pipeline':
        from radio_gs.querying.source_accepted_category import category_scope_decision
        scope=json.loads((root/'bounded_fallback_withheld/report.json').read_text())
        admission=json.loads((Path(data['base_field']).parent/'report.json').read_text())
        data['kernel_category']=dict(use_scope=category_scope_decision(scope,admission)['use_scope'],competition=competition)
    info = json.loads((root/'teachers.json').read_text())
    auth = json.loads((root/'responsibility.json').read_text())
    evidence = load(root/'caption_region_evidence.pt')
    parent = load(data['base_field'])
    model = ElementRetrievalField(**parent['architecture']).cuda()
    model.load_state_dict(parent['state_dict'])
    model.requires_grad_(False).eval()
    attach_kernel_native(model, data)
    n = len(model.observed)
    masks = evidence['masks'].flatten(1).bool()
    views = evidence['proposal_views']
    records = evidence['records']
    usable = [i for i,r in enumerate(records) if r['confidence'] >= .7 and masks[i].sum() >= 20]
    occurrences = {}
    for i in usable:
        for qi in records[i]['phrase_indices']:
            occurrences.setdefault(qi, set()).add(int(views[i]))
    source_views = [info['frames'].index(f) for f in info['train_frames']]
    held = set(source_views[::5])
    query_ids = sorted(q for q,v in occurrences.items() if len(v-held) >= 2)
    q = F.normalize(evidence['text_embeddings'][query_ids].float().cuda(), dim=1)
    query_index = {qi:j for j,qi in enumerate(query_ids)}
    base_logits=torch.cat([(model.logits(part,'text') if a.baseline=='pipeline' else model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(part)-model.native_query_thresholds['text'](part))) for part in q.split(24)],dim=1)
    baseline=base_logits.sigmoid()
    examples = []
    pixels = masks.shape[1]
    for vi in source_views:
        ids = [i for i in usable if int(views[i]) == vi]
        if not ids:
            continue
        queries = sorted({qi for i in ids for qi in records[i]['phrase_indices'] if qi in query_index})
        columns = torch.tensor([query_index[qi] for qi in queries], device='cuda')
        local_masks = masks[ids].float().cuda()
        descriptors = F.normalize(evidence['native_descriptors'][ids].float().cuda(), dim=1)
        affinity = descriptors @ q[columns].T
        truth, known = [], []
        for j,qi in enumerate(queries):
            positives = [k for k,i in enumerate(ids) if qi in records[i]['phrase_indices']]
            positive = local_masks[positives].bool().any(0)
            cutoff = affinity[positives,j].quantile(.25) - .03
            eligible = affinity[:,j] < cutoff
            if a.negative_policy=='disjoint_regions':
                # Overlapping masks can describe an enclosing object or a part.
                # Their non-overlap is not evidence of semantic exclusion.
                eligible &= ~(local_masks.bool() & positive[None]).any(1)
            negative = local_masks[eligible].bool().any(0) & ~positive
            truth.append(positive)
            known.append(positive | negative)
        target = torch.stack(truth, dim=1).float()
        known = torch.stack(known, dim=1).float()
        hit = load(root/auth['views'][vi]['relative_path'])
        pid,gid = hit['pixel_ids'].long().cuda(),hit['gaussian_ids'].long().cuda()
        weight = hit['base_weights'].float().cuda()
        total = torch.zeros(pixels,device='cuda').scatter_add_(0,pid,weight)
        weight *= model.observed[gid]
        mass = torch.zeros_like(total).scatter_add_(0,pid,weight)
        visible = (mass/total.clamp_min(1e-8)>.5)&(total>1e-8)
        known *= visible[:,None]
        op = torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(pixels,n)).coalesce()
        examples.append(dict(view=vi,op=op,adjoint=op.T.coalesce(),columns=columns,target=target,known=known,visible=visible,mass=mass))
        print(a.key,'source text extent',vi,'queries',len(queries),'held',vi in held,flush=True)
    pre_state = baseline.clone()
    value = baseline.clone()
    denominator = torch.zeros_like(value)
    visible_denominator = torch.zeros_like(value)
    if a.evidence_policy=='consistent_views':
        positive_views=torch.zeros_like(value,dtype=torch.uint8)
        negative_views=torch.zeros_like(value,dtype=torch.uint8)
    for e in examples:
        if e['view'] not in held:
            mass=torch.sparse.mm(e['adjoint'],e['known'])
            denominator[:,e['columns']] += mass
            visible_mass=torch.sparse.mm(e['adjoint'],e['visible'].float()[:,None])
            visible_denominator[:,e['columns']] += visible_mass
            if a.evidence_policy=='consistent_views':
                positive_mass=torch.sparse.mm(e['adjoint'],e['known']*e['target'])
                fraction=positive_mass/mass.clamp_min(1e-8)
                positive_views[:,e['columns']] += ((fraction>=.8)&(mass>=.25)).byte()
                negative_views[:,e['columns']] += ((fraction<=.2)&(mass>=.25)).byte()
    consistency=None
    if a.evidence_policy=='consistent_views':
        consistency=((positive_views>=2)&(negative_views==0))|((negative_views>=2)&(positive_views==0))
        consistency_report=dict(confirmed_element_query_pairs=int(consistency.sum()),contradictory_element_query_pairs=int(((positive_views>0)&(negative_views>0)).sum()),strongly_contradictory_pairs=int(((positive_views>=2)&(negative_views>=2)).sum()),observation_supported_pairs=int((denominator>0).sum()),minimum_independent_views=2,policy='pure same-sign confirmations; opposite evidence or fewer than two confirmations abstain to parent')
        print(a.key,consistency_report,flush=True)
    if a.carrier=='scopes':
        graph=data['scope_hypergraph']
        membership=torch.sparse_coo_tensor(graph['indices'].cuda(),graph['weights'].cuda(),graph['shape']).coalesce()
        transpose=membership.T.coalesce()
        degree=torch.sparse.sum(membership,dim=1).to_dense()
        majorizer=torch.sparse.mm(transpose,denominator*degree[:,None]).clamp_min(1e-8)
        prior_majorizer=(torch.sparse.mm(transpose,visible_denominator*degree[:,None]).clamp_min(1e-8)
                         if a.prior_domain=='visible' else majorizer)
        spatial_coefficients=torch.zeros((graph['shape'][1],len(q)),device='cuda')
    history=[]
    def measure(step):
        row={'step':step}
        for split in ['train','held_source_views']:
            rows=[]
            for e in examples:
                if (e['view'] in held) != (split=='held_source_views'):
                    continue
                pred=torch.sparse.mm(e['op'],value[:,e['columns']])
                parent_pred=torch.sparse.mm(e['op'],baseline[:,e['columns']])
                rows.extend(episode_rows(extent_measures(pred,e,parent_pred),e['view'],e['columns']))
            row[split]=aggregate(rows)
        history.append(row);print(a.key,row,flush=True)
    measure(0)
    for step,pre_state,spatial_coefficients,value in extent_training_states(
            base_logits,examples,held,denominator,visible_denominator,a.steps,
            a.prior,a.loss,a.readout,a.prior_domain,consistency,a.carrier,
            membership if a.carrier=='scopes' else None,
            transpose if a.carrier=='scopes' else None,
            majorizer if a.carrier=='scopes' else None,
            prior_majorizer if a.carrier=='scopes' else None,
            spatial_coefficients if a.carrier=='scopes' else None):
        if step%20==0:measure(step)
    # Native kernel interpolation has no dependence on other runtime queries.
    concentration=20.
    gram=torch.exp(concentration*(q@q.T-1))
    delta=torch.sparse.mm(membership,spatial_coefficients) if a.carrier=='scopes' else pre_state-baseline
    if consistency is not None:delta*=consistency
    # Match serialized precision and deployment anchor normalization.
    anchors=q.half()
    normalized_anchors=F.normalize(anchors.float(),dim=1)
    gram=torch.exp(concentration*(normalized_anchors@normalized_anchors.T-1))
    source_kernel=torch.exp(concentration*(normalized_anchors@q.T-1))
    delta=delta.half().float()
    choices=[]
    support=denominator>0
    for ridge in [.1,1.,10.]:
        inverse=torch.linalg.inv(gram+ridge*torch.eye(len(q),device='cuda'))
        correction=delta@(inverse@source_kernel)
        if a.interpolation=='conditional_evidence':
            correction=(delta@source_kernel)/(support.float()@source_kernel+ridge)
        fitted=baseline+correction
        if a.readout=='odds':fitted=odds_probability(base_logits,correction)
        rows=[]
        for e in examples:
            if e['view'] not in held:continue
            pred=torch.sparse.mm(e['op'],fitted[:,e['columns']].clamp(0,1))
            parent_pred=torch.sparse.mm(e['op'],baseline[:,e['columns']])
            rows.extend(episode_rows(extent_measures(pred,e,parent_pred),e['view'],e['columns']))
        choices.append(dict(ridge=ridge,**aggregate(rows),episodes=rows))
    best=select_ridge(choices,a.selection_rule)
    data['source_kernel_extent']=dict(anchors=anchors.cpu(),inverse=torch.linalg.inv(gram+best['ridge']*torch.eye(len(q),device='cuda')).cpu(),concentration=concentration,ridge=best['ridge'],readout=a.readout)
    if a.carrier=='scopes' and consistency is None and a.interpolation!='conditional_evidence':data['source_kernel_extent']['scope_coefficients']=spatial_coefficients.half().cpu()
    else:data['source_kernel_extent']['residual']=delta.half().cpu()
    data['source_kernel_extent']['interpolation']=a.interpolation
    if a.interpolation=='conditional_evidence':data['source_kernel_extent']['evidence_support']=support.cpu()
    data['method']='source-rendered text extents with native kernel interpolation'
    out=Path(a.output)/a.key;out.mkdir(parents=True,exist_ok=True)
    torch.save(data,out/'field.pt')
    if a.dual_selection:
        for rule in ['legacy_visible_iou','partial_known_brier']:
            selected=select_ridge(choices,rule)
            state=dict(data['source_kernel_extent'],ridge=selected['ridge'],selection_rule=rule,
                       inverse=torch.linalg.inv(gram+selected['ridge']*torch.eye(len(q),device='cuda')).cpu())
            arm=Path(a.output)/rule/a.key;arm.mkdir(parents=True,exist_ok=False)
            torch.save(dict(data,source_kernel_extent=state),arm/'field.pt')
            (arm/'selection.json').write_text(json.dumps(clean_json(dict(rule=rule,selected=selected,shared_training=str(out/'source_training.pt'))),indent=2,allow_nan=False))
    if a.save_training:
        torch.save(dict(residual=delta.half().cpu(),steps=a.steps,baseline=baseline.half().cpu(),base_logits=base_logits.cpu(),query=q.cpu(),examples=[{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in e.items()} for e in examples],held=sorted(held)),out/'source_training.pt')
    report=dict(key=a.key,steps=a.steps,loss=a.loss,baseline=a.baseline,carrier=a.carrier,prior=a.prior,source_queries=len(query_ids),phrases=[evidence['phrases'][i] for i in query_ids],history=history,kernel_choices=choices,selected=best,held_source_frames=[info['frames'][i] for i in sorted(held)],geometry_frozen=True,native_coordinates_frozen=True,teacher_masks_are_pseudo_labels=True,missing_detections_unknown=True,negative_policy='other confident source masks with lower native affinity than own positive quartile minus .03',benchmark_query_or_gt_used=False,validation_scope='held for extent fitting; original native map already used these source views')
    if consistency is not None:report['view_consistency']=consistency_report
    report['readout']=a.readout
    report['training_state']='pre-readout probability; residual=pre_state-baseline'
    report['odds_gradient']='straight-through identity approximation, not exact chain rule'
    report['prior_domain']=a.prior_domain
    report.update(negative_policy=a.negative_policy,interpolation=a.interpolation,selection_metric=a.selection_rule,unknown_tie_diagnostic='reported only; numerical ties within 1e-6 prefer larger ridge')
    (out/'training.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False)+'\n')
    print(a.key,'complete',best,flush=True)


if __name__=='__main__':main()
