"""Independent current-mask resampling, preparation, optics and target metrics."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
from scipy import sparse
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.geometry import projection,selected_alpha
from radio_gs.query_field.extent_decision import render_binary
from scripts.research.run_query_field_reference_extent import ROOT,PREVIOUS,ARMS,design
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes


def area_reference(path,hw):
    raw=np.asarray(Image.open(path));raw=np.any(raw>127,axis=2) if raw.ndim==3 else raw>127
    h,w=hw;rh,rw=raw.shape
    integral=np.pad(raw.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
    y0=np.floor(np.arange(h)*rh/h).astype(int);y1=np.ceil(np.arange(1,h+1)*rh/h).astype(int)
    x0=np.floor(np.arange(w)*rw/w).astype(int);x1=np.ceil(np.arange(1,w+1)*rw/w).astype(int)
    value=(integral[y1[:,None],x1]-integral[y0[:,None],x1]-integral[y1[:,None],x0]+integral[y0[:,None],x0])
    # Match the declared float32 adaptive-area evaluation order: divide by
    # height, then width. Integer sums are independently reconstructed above.
    return value.astype(np.float32)/(y1-y0)[:,None].astype(np.float32)/(x1-x0).astype(np.float32)


def objective(alpha,selected,state):
    scope=state['scope'];y=scope['target'].double().numpy();p=scope['positive'].numpy();n=scope['negative'].numpy();known=p|n
    cutoff=state['decision_cutoff'];a=alpha.astype(np.float64).clip(1e-6,1-1e-6)
    probability=a*(1-cutoff)/(a*(1-cutoff)+cutoff*(1-a));b=probability.clip(1e-5,1-1e-5)
    loss=-y*np.log(b)-(1-y)*np.log1p(-b)
    data=.5*((loss*p).sum()/max(p.sum(),1)+(loss*n).sum()/max(n.sum(),1))
    data+=1-(2*(probability*y*known).sum()+1)/((probability*known).sum()+(y*known).sum()+1)
    cost=.1*state['cost'].double().numpy()@selected
    return float(cost+float(state['confidence'][0])*data)


def mask_metrics(pred,truth):
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
    return dict(iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))


@torch.no_grad()
def audit(index):
    d=design();j=d['jobs'][index];cfg=read(j['configuration']['path']);name=j['name']
    read(ROOT/'predictions_complete.json');read(ROOT/'scoring_complete.json')
    evaluation=read(ROOT/'predictions'/name/'evaluation.json');scoring=read(j['scoring']['path'])
    check(dict(path=scoring['gt_mask'],sha256=evaluation['gt_sha256']))
    truth=np.asarray(Image.open(scoring['gt_mask']));truth=np.any(truth>127,axis=2) if truth.ndim==3 else truth>127
    records={};data={};states={};metrics={};pix={};h,w=cfg['projection_hw']
    for job,arm in zip(cfg['jobs'],ARMS):
        p=ROOT/'predictions'/name/job['name'];r=read(p/'prediction.json')
        assert r['configuration']==cfg and r['job']==job and not r['target_labels_opened']
        assert not r['runtime_source_access'] and not r['reference_rgb_opened'] and not r['negative_used_for_membership']
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
        data[arm]=torch.load(p/'predictions.pt',weights_only=True);records[arm]=r
        assert torch.equal(data[arm]['membership'],data[arm]['selected'].float())
        selected=cv2.resize(data[arm]['response'].numpy(),truth.shape[::-1],interpolation=cv2.INTER_LINEAR)>cfg['alpha_threshold']
        normalized=cv2.resize(data[arm]['retained_response'].numpy(),truth.shape[::-1],interpolation=cv2.INTER_LINEAR)
        metric=mask_metrics(selected,truth);response=mask_metrics(normalized>=cfg['alpha_threshold'],truth)
        row=next(x for x in evaluation['results'] if x['job']==job['name'])
        for k,m in [('iou','foreground_iou'),('precision','precision'),('recall','recall')]:assert abs(metric[k]-row[m])<1e-12
        assert abs(response['iou']-row['retained_response']['foreground_iou'])<1e-12
        score=normalized.ravel();labels=truth.ravel();order=np.argsort(-score,kind='stable')
        values=score[order];ends=np.r_[np.flatnonzero(values[1:]!=values[:-1]),len(score)-1]
        tp=labels[order].astype(np.float64).cumsum()[ends];ap=float(np.sum(np.diff(np.r_[0.,tp])*tp/(ends+1))/labels.sum())
        assert abs(ap-row['retained_response']['average_precision'])<1e-12
        metrics[arm]=dict(**metric,response_iou=response['iou'],ap=ap);pix[arm]=selected
        if arm!='composed':
            check(r['extent_state']);states[arm]=torch.load(r['extent_state']['path'],weights_only=True)
    old=torch.load(PREVIOUS/'predictions'/name/(name+'__admission')/'predictions.pt',weights_only=True)
    assert torch.equal(old['selected'],data['composed']['selected'])
    for key in ['membership','response','retained_response']:assert float((old[key]-data['composed'][key]).abs().max())<=1e-6
    assert all(r['reference_mask_receipt']==records['composed']['reference_mask_receipt'] for r in records.values())
    field,_=load_field(cfg['jobs'][0]['field'],'cuda')
    fixture=torch.load(cfg['jobs'][0]['camera_fixture'],weights_only=True);camera=SimpleNamespace(**fixture['reference'])
    assert fixture['world_units_per_metre']==field.world_units_per_metre
    domain=field.member_available.cpu().numpy();count=len(domain)
    ref=area_reference(cfg['jobs'][0]['reference_mask'],(h,w));target=ref.flatten()
    geo=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,
                        get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
    raw,_,_,_=projection(geo,camera,h,w,field.world_units_per_metre)
    row,col=raw.indices().cpu().numpy();weight=raw.values().double().cpu().numpy()
    total=np.bincount(row,weights=weight,minlength=h*w)
    mass=np.bincount(row,weights=weight*domain[col],minlength=h*w)
    valid=(mass>1e-8)&(mass/np.maximum(total,1e-8)>.5)
    op=sparse.coo_matrix((weight*domain[col]/np.maximum(mass[row],1e-8),(row,col)),shape=(h*w,count)).tocsr()
    from gsplat.cuda._wrapper import fully_fused_projection
    k=camera.intrinsic.to(field.xyz).clone();k[0]*=w/camera.width;k[1]*=h/camera.height
    radii,_,_,conics,_=fully_fused_projection(field.xyz,None,field.quaternions,field.scales,
        camera.world_to_camera.to(field.xyz)[None],k[None],w,h,near_plane=.001*field.world_units_per_metre,far_plane=100*field.world_units_per_metre)
    conic=conics[0].double().cpu().numpy();eligible=(radii[0].cpu().numpy()>0)&domain
    footprint=np.zeros(count);footprint[eligible]=2*np.pi*field.opacities.flatten().cpu().numpy()[eligible]/np.sqrt(np.maximum(conic[eligible,0]*conic[eligible,2]-conic[eligible,1]**2,1e-12))
    probes={};changes={};prior=data['composed']['selected'].numpy()
    for arm,state in states.items():
        assert np.array_equal(state['domain'].numpy(),domain)
        assert np.array_equal(state['prior'].numpy(),prior.astype(np.float32))
        assert np.array_equal(state['reference'].numpy(),ref)
        assert torch.equal(state['selected'],data[arm]['selected'])
        assert not state['selected'][~state['domain']].any()
        assert state['decision_cutoff']==d['decision_cutoff'][arm]
        scope=state['scope'];lift=state['lift'];opt=state['optical']
        assert np.array_equal(lift['valid'].numpy(),valid)
        assert np.array_equal(scope['target'].numpy(),target)
        assert np.array_equal(scope['positive'].numpy(),valid&(target>0))
        assert np.array_equal(scope['negative'].numpy(),valid&(target==0))
        actual=sparse.coo_matrix((lift['weight'].double().numpy(),lift['index'].numpy()),shape=op.shape).tocsr()
        delta=actual-op;lift_error=float(np.max(np.abs(delta.data))) if delta.nnz else 0.
        assert lift_error<=2e-6,(name,arm,'lift',lift_error)
        footprint_error=float(np.max(np.abs(lift['footprint'].numpy()-footprint)/np.maximum(footprint,1e-8)))
        assert footprint_error<=2e-5,(name,arm,'footprint',footprint_error)
        active=bool((valid&(target>0)).any());assert float(state['confidence'][0])==float(active)
        if arm=='shared_lift' and active:
            known=op.T@valid.astype(float);positive=op.T@(valid*target)
            initial=np.where(known>1e-8,positive/np.maximum(known,1e-8),prior)
        else:initial=prior.astype(float)
        initial_error=float(np.max(np.abs(initial-state['initial'].numpy())))
        cost=footprint/max(float(footprint@initial),1e-8) if active else np.zeros(count)
        cost_error=float(np.max(np.abs(cost-state['cost'].numpy())/np.maximum(cost,1e-8)))
        assert initial_error<=2e-6 and cost_error<=2e-5,(name,arm,'preparation',initial_error,cost_error)
        info=records[arm]['diagnostics']['extent_decision'];endpoints={}
        for endpoint,bits in [('initial',state['initial'].numpy()>.5),('final',state['selected'].numpy())]:
            compiled=render_binary(opt['pointer'].numpy(),opt['index'].numpy(),opt['alpha'].numpy(),bits)
            rendered=selected_alpha(field,torch.from_numpy(bits).cuda(),camera,h,w,field.world_units_per_metre).cpu().numpy()
            error=float(np.max(np.abs(compiled-rendered)));bits_diff=int(((compiled>state['decision_cutoff'])!=(rendered>state['decision_cutoff'])).sum())
            assert error<=1e-4 and bits_diff==0,(name,arm,endpoint,'optics',error,bits_diff)
            value=objective(compiled,bits,state);physical=objective(rendered,bits,state)
            if active:
                expected=info['energy_trace'][0 if endpoint=='initial' else -1]
                assert abs(value-expected)<1e-7 and abs(physical-expected)<2e-6,(name,arm,endpoint,value,physical,expected)
            binary=rendered>.5;intersection=float(target[binary].sum())
            endpoints[endpoint]=dict(objective=value,renderer_objective=physical,alpha_max_error=error,threshold_bit_difference=bits_diff,
                reference_soft_iou=intersection/max(float(binary.sum()+target.sum()-intersection),1e-12))
        probes[arm]=dict(lift_max_absolute=lift_error,footprint_max_relative=footprint_error,
                        initial_max_absolute=initial_error,cost_max_relative=cost_error,endpoints=endpoints)
        added=pix[arm]&~pix['composed'];removed=pix['composed']&~pix[arm]
        changes[arm]={k:int(v) for k,v in dict(added_tp=(added&truth).sum(),added_fp=(added&~truth).sum(),
                    removed_tp=(removed&truth).sum(),removed_fp=(removed&~truth).sum()).items()}
    # Literal and task-aware conditions differ only in decision cutoff.
    for key in ['initial','cost','prior','reference','domain']:
        assert torch.equal(states['shared_prior'][key],states['shared_literal'][key]),(name,key)
    result=dict(passed=True,case=name,metrics=metrics,probes=probes,pixel_changes=changes,
        same_current_input=True,baseline_selection_exact=True,all_binary_memberships=True,
        source_access=False,new_target_scores_selected_model=False)
    write(ROOT/'audits'/(name+'.json'),result);print('audited',name,flush=True)


def run(gpus):
    d=design()
    for r in read(ROOT/'predictions_complete.json')['records']+read(ROOT/'scoring_complete.json')['records']:check(r)
    lanes(list(range(8)),gpus,lambda i,g:execute(['-m','scripts.research.audit_query_field_reference_extent','case','--index',str(i)],ROOT/'logs'/('audit_'+d['jobs'][i]['name']+'.log'),g))
    cases={j['name']:read(ROOT/'audits'/(j['name']+'.json')) for j in d['jobs']}
    result=dict(passed=all(c['passed'] for c in cases.values()),cases=cases,queries=8,conditions=4,
        predictions=32,summary=receipt(ROOT/'summary.json'),design=receipt(ROOT/'design.json'),
        code=receipt(__file__),whole_five_task=False,model_promotion=False,raw_scribble=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-reference-extent-audit.json',result)
    print('PASS all eight independent reference extent audits',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['case','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,2,3,4')
    a=p.parse_args();torch.set_num_threads(4)
    audit(a.index) if a.stage=='case' else run(list(map(int,a.gpus.split(','))))
