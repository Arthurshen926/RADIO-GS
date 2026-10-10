"""Join complete post-prediction diagnostics, keeping pseudo-label scope explicit."""
import json
from pathlib import Path
from radio_gs.query_field.storage import sha256


def main():
    scenes=['figurines','ramen','teatime','waldo_kitchen'];studies={};reference_labels={}
    for study in ['joint_exclusion_v1','joint_exclusion_prior_v1']:
        root=Path('output/query_field_20261010')/study;totals={};domain={};source_iou={};pairs=0;positive=negative=0;receipts=[]
        for scene in scenes:
            source=root/'diagnostics'/('source_'+scene+'.json');p=json.loads(source.read_text())
            fields=root/'diagnostics'/('domain_'+scene+'.json');d=json.loads(fields.read_text())
            assert p['target_labels_used']==d['target_labels_opened']==False and not p['inference_inputs_changed']
            labels=[(r['query'],r['frame'],r['positive'],r['negative']) for r in p['records']]
            if study=='joint_exclusion_v1':reference_labels[scene]=labels
            else:assert labels==reference_labels[scene]
            pairs+=p['query_view_pairs'];positive+=p['positive'];negative+=p['negative']
            for a,t in p['totals'].items():
                source_iou.setdefault(a,{})[scene]=dict(pooled=t['positive_covered']/max(p['positive']+t['negative_covered'],1),
                    query_view_mean=sum(r['arms'][a]['positive_covered']/max(r['positive']+r['arms'][a]['negative_covered'],1) for r in p['records'])/len(p['records']))
                merged=totals.setdefault(a,{k:0 for k in t})
                for k,v in t.items():merged[k]+=v
                assert t['positive_coverage_lost']==t['sampled_positive_lost']+t['unsampled_positive_lost']
            for r in d['rows']:
                for k,v in r.items():
                    if k!='text':domain[k]=domain.get(k,0)+v
            receipts += [dict(path=str(f),sha256=sha256(f)) for f in [source,fields]]
        cold=json.loads((root/'validation/cold_figurines.json').read_text())
        assert cold['passed'] and cold['actual_sets_exact'] and not cold['denied_attempts']
        studies[study]=dict(query_view_pairs=pairs,known_positive_occurrences=positive,known_negative_occurrences=negative,
            full_source=totals,source_partial_known_iou=source_iou,source_partial_known_iou_macro={a:{k:sum(v[k] for v in scenes.values())/4 for k in ["pooled","query_view_mean"]} for a,scenes in source_iou.items()},member_domain=domain,receipts=receipts,cold_query_api=cold,
            source_labels_are_partial_pseudo_labels=True,not_benchmark_error_rates=True)
    report=dict(passed=True,studies=studies,matching_source_conditions=True,thresholds_changed=False,
        construction_conditions=64,benchmark_queries_per_study=68,whole_five_task=False,code_sha256=sha256(__file__))
    Path('docs/experiments/2026-10-10-joint-exclusion-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in r.items() if k not in ['receipts','cold_query_api']} for s,r in studies.items()},indent=2))

if __name__=='__main__':main()
