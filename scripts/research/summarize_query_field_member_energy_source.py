"""Paired full-source diagnostics with unchanged pseudo labels and old controls."""
from scripts.research.run_query_field_member_energy import ROOT,PARENT,ARMS,read,write,receipt


def main():
    per={};inputs=[]
    for job in read(ROOT/'prediction_config.json')['jobs']:
        path=ROOT/'diagnostics'/('source_'+job['scene']+'.json');new=read(path)
        old=read(PARENT/'diagnostics'/('source_'+job['scene']+'.json'))
        for k in ['query_view_pairs','positive','negative']:assert new[k]==old[k]
        assert len(new['records'])==len(old['records'])
        for a,b in zip(new['records'],old['records']):
            for k in ['query','text','frame','positive','negative']:assert a[k]==b[k]
            for arm in ARMS[:3]:assert a['arms'][arm]==b['arms'][arm]
        totals=new['totals']
        for arm in ARMS:
            totals[arm]['positive_coverage_gained']=totals[arm]['positive_covered']-totals['semantic_initial']['positive_covered']+totals[arm]['positive_coverage_lost']
        per[job['scene']]=dict(query_view_pairs=new['query_view_pairs'],positive=new['positive'],negative=new['negative'],totals=totals,
            partial_known_iou={a:totals[a]['positive_covered']/max(new['positive']+totals[a]['negative_covered'],1) for a in ARMS})
        inputs.append(receipt(path))
    result=dict(complete=True,per_scene=per,source_labels_unchanged=True,old_controls_render_exact=True,
        partial_pseudo_labels_not_GT=True,occurrences_not_independent_samples=True,inference_inputs_changed=False,
        query_view_pairs=sum(x['query_view_pairs'] for x in per.values()),
        positive=sum(x['positive'] for x in per.values()),negative=sum(x['negative'] for x in per.values()),
        totals={a:{k:sum(x['totals'][a][k] for x in per.values()) for k in next(iter(per.values()))['totals'][a]} for a in ARMS},
        partial_known_iou_scene_macro={a:sum(x['partial_known_iou'][a] for x in per.values())/4 for a in ARMS},inputs=inputs)
    write(ROOT/'diagnostics/summary.json',result)
    write('docs/experiments/2026-10-11-member-energy-source-results.json',result)
    print({k:v for k,v in result.items() if k not in ['per_scene','inputs']},flush=True)

if __name__=='__main__':main()
