"""Independent observation/operator replay and all target metrics for 16 sets."""
import argparse
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
from scipy import sparse
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.geometry import projection,selected_alpha
from radio_gs.query_field.extent_decision import render_binary
from scripts.research.run_query_field_reference_coupled import ROOT,PREVIOUS,ARMS,design
from scripts.research.run_query_field_relations import read,write,receipt,check,execute,lanes
from scripts.research.audit_query_field_reference_extent import area_reference,mask_metrics


def data_loss(p,y,positive,negative):
    p=np.clip(p.astype(float),1e-6,1-1e-6);b=np.clip(p,1e-5,1-1e-5);known=positive|negative
    ce=-y*np.log(b)-(1-y)*np.log1p(-b)
    return .5*((ce*positive).sum()/max(positive.sum(),1)+(ce*negative).sum()/max(negative.sum(),1))+1-(2*(p*y*known).sum()+1)/((p*known).sum()+(y*known).sum()+1)


def soft_target_iou(p,y):
    s=p>.5;inter=float(y[s].sum())
    return inter/max(float(s.sum()+y.sum()-inter),1e-12)


@torch.no_grad()
def audit(index):
    d=design();j=d['jobs'][index];name=j['name'];cfg=read(j['configuration']['path'])
    read(ROOT/'predictions_complete.json');read(ROOT/'scoring_complete.json')
    evaluation=read(ROOT/'predictions'/name/'evaluation.json');old_eval=read(PREVIOUS/'predictions'/name/'evaluation.json')
    assert evaluation['gt_sha256']==old_eval['gt_sha256']
    scoring=read(j['scoring']['path']);check(dict(path=scoring['gt_mask'],sha256=evaluation['gt_sha256']))
    truth=np.asarray(Image.open(scoring['gt_mask']));truth=np.any(truth>127,axis=2) if truth.ndim==3 else truth>127
    old=read(PREVIOUS/'predictions'/name/(name+'__shared_prior')/'prediction.json');check(old['extent_state'])
    old_state=torch.load(old['extent_state']['path'],weights_only=True)
    field,_=load_field(cfg['jobs'][0]['field'],'cuda');fixture=torch.load(cfg['jobs'][0]['camera_fixture'],weights_only=True)
    camera=SimpleNamespace(**fixture['reference']);h,w=cfg['projection_hw']
    geo=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,
        get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
    raw,_,_,_=projection(geo,camera,h,w,field.world_units_per_metre)
    row,col=raw.indices().cpu().numpy();weight=raw.values().double().cpu().numpy();domain=field.member_available.cpu().numpy()
    mass=np.bincount(row,weights=weight*domain[col],minlength=h*w)
    total=np.bincount(row,weights=weight,minlength=h*w);valid=(mass>1e-8)&(mass/np.maximum(total,1e-8)>.5)
    op=sparse.coo_matrix((weight*domain[col]/np.maximum(mass[row],1e-8),(row,col)),shape=(h*w,len(domain))).tocsr()
    target=area_reference(cfg['jobs'][0]['reference_mask'],(h,w)).ravel();p=valid&(target>0);n=valid&(target==0)
    result={}
    for arm,job in zip(ARMS,cfg['jobs']):
        directory=ROOT/'predictions'/name/job['name'];r=read(directory/'prediction.json')
        assert r['configuration']==cfg and r['job']==job and not r['target_labels_opened']
        assert not r['runtime_source_access'] and not r['reference_rgb_opened']
        assert r['reference_mask_receipt']==old['reference_mask_receipt']
        check(dict(path=str(directory/'predictions.pt'),sha256=r['predictions_sha256']));check(r['extent_state'])
        values=torch.load(directory/'predictions.pt',weights_only=True);state=torch.load(r['extent_state']['path'],weights_only=True)
        assert torch.equal(values['selected'],state['selected']) and torch.equal(values['membership'],state['selected'].float())
        for k in ['prior','initial','reference','domain']:assert torch.equal(state[k],old_state[k]),(name,arm,k)
        assert np.array_equal(state['domain'].numpy(),domain) and not state['selected'][~state['domain']].any()
        assert torch.equal(state['lift']['footprint'],old_state['lift']['footprint'])
        cost_error=float(np.max(np.abs(state['cost'].numpy()-old_state['cost'].numpy())/np.maximum(old_state['cost'].numpy(),1e-8)))
        assert cost_error<=2e-6,(name,arm,'bound cost',cost_error)
        assert state['decision_cutoff']==.5 and state['observation_weight']==d['observation_weight'][arm]
        scope=state['scope'];assert np.array_equal(scope['target'].numpy(),target)
        assert np.array_equal(scope['positive'].numpy(),p) and np.array_equal(scope['negative'].numpy(),n)
        assert float(state['confidence'][0])==1. and float(scope['confidence'])==1.
        lift=state['lift'];actual=sparse.coo_matrix((lift['weight'].double().numpy(),lift['index'].numpy()),shape=op.shape).tocsr()
        diff=actual-op;lift_error=float(np.abs(diff.data).max()) if diff.nnz else 0.
        assert lift_error<=2e-6 and np.array_equal(lift['valid'].numpy(),valid)
        opt=state['optical']
        for k in ['pointer','index','alpha','atom']:assert torch.equal(opt[k],old_state['optical'][k]),(name,arm,k)
        endpoints={};mix=state['observation_weight'];cost=state['cost'].double().numpy();y=target.astype(float)
        for endpoint,bits in [('initial',state['initial'].numpy()>.5),('final',state['selected'].numpy())]:
            compiled=render_binary(opt['pointer'].numpy(),opt['index'].numpy(),opt['alpha'].numpy(),bits)
            rendered=selected_alpha(field,torch.from_numpy(bits).cuda(),camera,h,w,field.world_units_per_metre).cpu().numpy()
            error=float(np.abs(compiled-rendered).max());bitdiff=int(((compiled>.5)!=(rendered>.5)).sum())
            assert error<=1e-4 and bitdiff==0,(name,arm,endpoint,'render',error,bitdiff)
            obs=op@bits;saved_obs=actual@bits
            value=.1*cost@bits+(1-mix)*data_loss(compiled,y,p,n)+mix*data_loss(saved_obs,y,p,n)
            independent=.1*cost@bits+(1-mix)*data_loss(rendered,y,p,n)+mix*data_loss(obs,y,p,n)
            expected=r['diagnostics']['extent_decision']['energy_trace'][0 if endpoint=='initial' else -1]
            assert abs(value-expected)<1e-7 and abs(independent-expected)<2e-6,(name,arm,endpoint,value,independent,expected)
            endpoints[endpoint]=dict(objective=value,independent_objective=independent,report_objective=expected,
                alpha_max_error=error,threshold_bit_difference=bitdiff,
                reference_set_iou=soft_target_iou(rendered,y),reference_observation_iou=soft_target_iou(obs[valid],y[valid]))
        selected=cv2.resize(values['response'].numpy(),truth.shape[::-1],interpolation=cv2.INTER_LINEAR)>.5
        response=cv2.resize(values['retained_response'].numpy(),truth.shape[::-1],interpolation=cv2.INTER_LINEAR)
        metrics=mask_metrics(selected,truth);response_metrics=mask_metrics(response>=.5,truth)
        report=next(v for v in evaluation['results'] if v['job']==job['name'])
        for k,m in [('iou','foreground_iou'),('precision','precision'),('recall','recall')]:assert abs(metrics[k]-report[m])<1e-12
        assert abs(response_metrics['iou']-report['retained_response']['foreground_iou'])<1e-12
        scores=response.ravel();labels=truth.ravel();order=np.argsort(-scores,kind='stable');values_sorted=scores[order]
        ends=np.r_[np.flatnonzero(values_sorted[1:]!=values_sorted[:-1]),len(scores)-1]
        tp=labels[order].astype(float).cumsum()[ends];ap=float((np.diff(np.r_[0.,tp])*tp/(ends+1)).sum()/labels.sum())
        assert abs(ap-report['retained_response']['average_precision'])<1e-12
        result[arm]=dict(passed=True,endpoints=endpoints,lift_max_error=lift_error,cost_binding_max_relative=cost_error,
            metrics=dict(**metrics,response_iou=response_metrics['iou'],ap=ap),record=receipt(directory/'prediction.json'))
    write(ROOT/'audits'/(name+'.json'),dict(passed=True,case=name,conditions=result,
        old_float64_footprint_gate_still_failed=name in {'fortress','trex'},prior_not_locked=True))
    print('audited coupled',name,flush=True)


def run(gpus):
    d=design()
    for r in read(ROOT/'predictions_complete.json')['records']+read(ROOT/'scoring_complete.json')['records']:check(r)
    lanes(list(range(8)),gpus,lambda i,g:execute(['-m','scripts.research.audit_query_field_reference_coupled','case','--index',str(i)],ROOT/'logs'/('audit_'+d['jobs'][i]['name']+'.log'),g))
    result=dict(passed=True,cases={j['name']:read(ROOT/'audits'/(j['name']+'.json')) for j in d['jobs']},
        predictions=16,endpoints=32,code=receipt(__file__),design=receipt(ROOT/'design.json'),
        previous_float64_footprint_failure_not_repaired=True,default_promoted=False,whole_five_task=False)
    write(ROOT/'audit.json',result);write('docs/experiments/2026-10-12-reference-coupled-audit.json',result)
    print('PASS 16 coupled/observation sets and 32 independent endpoints',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['case','run']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,2,3,4')
    a=p.parse_args();torch.set_num_threads(4)
    audit(a.index) if a.stage=='case' else run(list(map(int,a.gpus.split(','))))
