"""Summarize complete fixed-input development controls and separate diagnostics."""
import json
from pathlib import Path


def mean(rows):
    return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]} if rows else {}


def main():
    result={'development_only':True,'units':'percent; equal-scene means','methods':{}}
    for name,base,subdir in [
        ('inverse_with_veto','rendered_scopes_next37','pipeline'),
        ('inverse_propagation','rendered_scope_propagation_next37','pipeline'),
        ('native_scope_union_pilots','native_scope_union_next37','pipeline'),
        ('native_dictionary_and_inverse_text_scopes','native_rendered_scope_next37','pipeline_calibrated')]:
        root=Path('output')/base;rows=[]
        for path in sorted(root.glob('*/anchored/'+subdir+'/evaluation.json')):
            data=json.loads(path.read_text());metrics={}
            for arm,v in data['metrics'].items():
                metrics[arm]=({k:100*v[k]['miou'] for k in ['19','15','10']} if '19' in v else
                             {k:100*v[k] for k in ['iou','ap','localization']}|{'selected_iou':100*v['selected']['miou']})
            rows.append(dict(key=data['key'],metrics=metrics,path=str(path)))
        output={'scenes':rows,'cohorts':{}}
        for cohort in ['lerf','scannet']:
            selected=[r for r in rows if r['key'].startswith(cohort)]
            output['cohorts'][cohort]={'count':len(selected),'macro':{arm:mean([r['metrics'][arm] for r in selected]) for arm in ['retained_pipeline','hierarchical_pipeline']}}
        for kind,pattern in [('image','*/reference_image_evaluation.json'),('nvos','*/evaluation_*/evaluation.json')]:
            selected=[]
            for path in sorted(root.glob(pattern)):
                data=json.loads(path.read_text());metrics=data['macro' if kind=='image' else 'metrics']
                selected.append(dict(key=data.get('key',data.get('scene')),path=str(path),metrics={arm:{k:100*v[k] for k in ['iou','average_precision']} for arm,v in metrics.items() if arm in ['retained','trained','prompt_conditioned']}))
            output['cohorts'][kind]={'count':len(selected),'scenes':selected,'macro':{arm:mean([r['metrics'][arm] for r in selected if arm in r['metrics']]) for arm in ['retained','trained','prompt_conditioned']}}
        result['methods'][name]=output
    result['scope_diagnostics']={}
    for name,base in [('original','scope_complete_next36'),('inverse','rendered_scopes_next37'),('inverse_with_local_evidence','local_rendered_scopes_next37')]:
        rows={}
        for path in sorted((Path('output')/base).glob('*/hypothesis_retrieval.json')):
            data=json.loads(path.read_text());rows[data['scene']]={k:100*v for k,v in data['metrics'].items() if k.endswith('iou')}
        result['scope_diagnostics'][name]=rows
    result['source_membership_fit']={}
    for path in sorted(Path('output/rendered_scopes_next37').glob('*/membership_fit.json')):
        data=json.loads(path.read_text());count=sum(r['before']['pairs'] for r in data['batches'])
        result['source_membership_fit'][data['key']]={side:{metric:sum(r[side][metric]*r[side]['pairs'] for r in data['batches'])/count for metric in ['iou','mse']} for side in ['before','after']}
    result['local_proposals']={path.parent.name:json.loads(path.read_text())['retained_local_proposals'] for path in Path('output/local_rendered_scopes_next37').glob('*/construction.json')}
    path=Path('output/rendered_scopes_next37/summary.json');path.write_text(json.dumps(result,indent=2)+'\n')
    for name,data in result['methods'].items():print(name,{k:dict(count=v['count'],macro=v['macro']) for k,v in data['cohorts'].items()})
    print('source fit',result['source_membership_fit'])


if __name__=='__main__':main()
