"""Veto incompatible concept extents using only co-observed source mask scope.

Leave the supervised view out of image prototypes; split validation by query
phrase. Missing detections are unknown. The all-pairs control excludes self
names; hard pairs prefer nonself positives and use self names only if needed.
The field, extents and native query metric remain frozen.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def coobserved_scope(masks,present):
    flat=masks.flatten(1).float();intersection=flat@flat.T
    union=flat.sum(1)[:,None]+flat.sum(1)[None]-intersection
    known=present[:,None]&present[None,:]&(union>0)
    return intersection/union.clamp_min(1),known


def select_hard_scope_pairs(similarity,overlap,known,self_pair):
    """One nearest compatible and incompatible extent; unknowns never negative."""
    positive=known&(overlap>=.5)
    nonself=positive&~self_pair
    positive=torch.where(nonself.any(1)[:,None],nonself,positive)
    negative=known&(overlap<=.1)&~self_pair
    selected=torch.zeros_like(known)
    for eligible in (positive,negative):
        rows=eligible.any(1).nonzero().flatten()
        columns=similarity[rows].masked_fill(~eligible[rows],-torch.inf).argmax(1)
        selected[rows,columns]=True
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--selected-only',action='store_true');p.add_argument('--hard-pairs',action='store_true');p.add_argument('--root',type=Path);a=p.parse_args()
    if a.selected_only and a.hard_pairs:raise ValueError('choose one pair policy')
    torch.set_num_threads(4);torch.manual_seed(42)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'native_overlap_fallback96';r=json.loads((parent/'report.json').read_text())
    assert sha256_file(parent/'field.pt')==r['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
    ep=root/'caption_region_evidence.pt';er=json.loads(ep.with_suffix('.json').read_text())
    assert sha256_file(ep)==er['sha256'];e=torch.load(ep,weights_only=False)
    assert e['source_only'] and not e['benchmark_data_used'] and e['geometry_binding']==payload['geometry_binding']
    groups=json.loads((root/'grounded_semantic_transfer96/report.json').read_text())['groups']
    qids=torch.tensor([g['source_query_index'] for g in groups],device=a.device)
    query=F.normalize(e['text_embeddings'].float().to(a.device),dim=-1);native=F.normalize(e['native_descriptors'].float().to(a.device),dim=-1)
    masks=e['masks'].to(a.device).bool();views=e['proposal_views'].to(a.device);P=len(query);G=len(groups)
    text=query@query[qids].T
    selected=text.argmax(1)
    original_acceptance=payload['state_dict']['grounded_transfer.acceptance'].to(a.device)
    admitted=text.max(1).values>=original_acceptance[selected]
    all_images=torch.stack([native[g['regions']].mean(0) for g in groups])
    features=[];targets=[];phrases=[];records=[]
    for vi in sorted(views.unique().tolist()):
        union=torch.zeros((P,*masks.shape[-2:]),dtype=torch.bool,device=a.device)
        present=torch.zeros(P,dtype=torch.bool,device=a.device)
        for rid in torch.where(views==vi)[0].tolist():
            for pi in e['records'][rid]['phrase_indices']:
                union[pi]|=masks[rid];present[pi]=True
        overlap,known=coobserved_scope(union,present)
        image=[];available=[]
        for g in groups:
            ids=[i for i in g['regions'] if int(views[i])!=vi]
            available.append(bool(ids));image.append(native[ids].mean(0) if ids else torch.zeros_like(native[0]))
        visual=query@torch.stack(image).T
        known=known[:,qids]&torch.tensor(available,device=a.device)[None]
        # Keep the pair policy explicit; self-name fitting is not generalization.
        if a.hard_pairs:
            known=select_hard_scope_pairs(text,overlap[:,qids],known,torch.arange(P,device=a.device)[:,None]==qids[None])
        elif a.selected_only:
            known &= (torch.arange(G,device=a.device)[None]==selected[:,None])&admitted[:,None]
        else:known &= torch.arange(P,device=a.device)[:,None]!=qids[None]
        pi,gi=known.nonzero(as_tuple=True)
        features.append(torch.stack([text[pi,gi],visual[pi,gi]],1));targets.append((overlap[pi,qids[gi]]>=.5).float());phrases.append(pi)
        records.append(dict(view=vi,pairs=len(pi),positive=int((overlap[pi,qids[gi]]>=.5).sum())))
    x=torch.cat(features);y=torch.cat(targets);pi=torch.cat(phrases)
    unique=pi.unique().cpu();order=unique[torch.randperm(len(unique))];heldout=order[::5].to(a.device)
    valid=torch.isin(pi,heldout);train=~valid
    counts=dict(train_pairs=int(train.sum()),train_positive=int(y[train].sum()),validation_pairs=int(valid.sum()),validation_positive=int(y[valid].sum()))
    if not (y[train].sum()>0 and (1-y[train]).sum()>0 and y[valid].sum()>0 and (1-y[valid]).sum()>0):
        path=Path(f'paper/artifacts/{a.scene}_source_scope_gate_insufficient_20260916.json')
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(dict(status='no model exported; both classes required in train and phrase-heldout validation',counts=counts,source_only=True,benchmark_data_used=False,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
        print(counts,flush=True);return
    def balance(mask):
        yy=y[mask];return torch.where(yy>.5,.5/yy.mean(),.5/(1-yy.mean()))
    tw,vw=(torch.ones_like(y[train]),torch.ones_like(y[valid])) if a.selected_only else (balance(train),balance(valid))
    raw=torch.tensor([3.,3.],device=a.device,requires_grad=True);bias=torch.tensor(-3.,device=a.device,requires_grad=True)
    optimizer=torch.optim.Adam([raw,bias],lr=.05);history=[];best=float('inf');state=None
    for step in range(1200):
        logits=x[train]@F.softplus(raw)+bias
        loss=(F.binary_cross_entropy_with_logits(logits,y[train],reduction='none')*tw).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%100==0:
            with torch.no_grad():
                vl=x[valid]@F.softplus(raw)+bias;v=float((F.binary_cross_entropy_with_logits(vl,y[valid],reduction='none')*vw).mean())
                history.append(dict(step=step+1,validation_objective_bce=v,weights=F.softplus(raw).tolist(),bias=float(bias)))
                if v<best:best=v;state=(F.softplus(raw).detach().clone(),bias.detach().clone())
    arch=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],routing={'num_competitors':0,'scope_gate':True}))
    field=ElementRetrievalField(**arch);missing=field.load_state_dict(payload['state_dict'],strict=False)
    expected={'grounded_transfer.routing_center','grounded_transfer.routing_competitors','grounded_transfer.routing_scope_images','grounded_transfer.routing_scope_weights','grounded_transfer.routing_scope_bias'}
    assert set(missing.missing_keys)==expected and not missing.unexpected_keys
    field.grounded_transfer.routing_scope_images.copy_(all_images.cpu());field.grounded_transfer.routing_scope_weights.copy_(state[0].cpu());field.grounded_transfer.routing_scope_bias.copy_(state[1].cpu())
    saved=field.state_dict()
    for k,v in payload['state_dict'].items():assert torch.equal(v,saved[k]),k
    def metrics(selection):
        yy=y[selection].bool();pred=x[selection]@state[0]+state[1]>=0
        return dict(pairs=int(selection.sum()),positive=int(yy.sum()),positive_recall=float(pred[yy].float().mean()),negative_acceptance=float(pred[~yy].float().mean()))
    out=root/('hard_source_scope_gate96' if a.hard_pairs else 'selected_source_scope_gate96' if a.selected_only else 'source_scope_gate96');out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch,state_dict=saved),out/'field.pt')
    report=dict(schema='source_scope_gate.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=r['field_sha256'],source_only=True,benchmark_data_used=False,
        all_parent_tensors_unchanged=True,evidence_sha256=sha256_file(ep),heldout_phrases=[e['phrases'][i] for i in heldout.tolist()],
        source_training=metrics(train),source_validation=metrics(valid),history=history,views=records,
        pair_policy='nearest compatible and incompatible co-observed extents; prefer nonself positive; balanced BCE' if a.hard_pairs else 'actual source nearest-concept admitted pairs including self-name; natural-prior BCE' if a.selected_only else 'all nonself co-observed pairs; balanced BCE',
        source_validation_scope='query phrases held out of gate fitting; base field and image prototypes saw source images; no independent final test',
        target='IoU >= 0.5 between co-observed source phrase mask unions; absent observation unknown; see pair_policy for self-name admission',
        features='native query/prototype cosine and native query/mean source region image cosine; image prototype excludes supervised view',
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,report['source_training'],report['source_validation'],state[0].tolist(),float(state[1]),flush=True)


if __name__=='__main__':main()
