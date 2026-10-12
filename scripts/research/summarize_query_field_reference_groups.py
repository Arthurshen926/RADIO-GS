"""Matched partition comparison including unchanged-initializer performance."""
from itertools import combinations
from scripts.research.run_query_field_reference_groups import ROOT,ARMS,design
from scripts.research.run_query_field_reference_extent import ROOT as ORIGINAL
from scripts.research.run_query_field_relations import read,write,receipt
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();audit=read(ROOT/'audit.json');assert audit['passed']
    new=read(ROOT/'summary.json');old=read(ORIGINAL/'summary.json');arms=['composed']+ARMS
    per={'composed':old['per_case']['composed'],**new['per_case']}
    macro={'composed':old['macro']['composed'],**new['macro']}
    comparisons={a+'__'+b:{k:paired({s:v[k] for s,v in per[a].items()},
        {s:v[k] for s,v in per[b].items()},'NVOS '+k) for k in macro[a]} for a,b in combinations(arms,2)}
    modifications={}
    for arm in ARMS:
        rows=[new['diagnostics'][arm][j['name']]['diagnostics']['extent_decision'] for j in d['jobs']]
        modifications[arm]=dict(added_members=sum(r['added'] for r in rows),removed_members=sum(r['removed'] for r in rows),
            cases_with_any_change=sum(r['added']+r['removed']>0 for r in rows),
            mean_final_objective=sum(r['energy_trace'][-1] for r in rows)/len(rows))
    reference={a:{k:sum(c['conditions'][a]['endpoints']['final'][k] for c in audit['cases'].values())/8
        for k in ['reference_set_iou','reference_observation_iou']} for a in ARMS}
    result=dict(macro=macro,per_case=per,comparisons=comparisons,member_modifications=modifications,
        reference_agreement=reference,primary='parts',old_composed_control_rerun=False,
        free_control_rerun_exact=all(r['selected_exact'] for r in read(ROOT/'predictions_complete.json')['baseline']),
        development_cases_not_heldout=True,whole_five_task=False,default_promoted=False,
        spatial_locality_not_separately_controlled=True,identity_truth_used=False,
        summary=receipt(ROOT/'summary.json'),old_summary=receipt(ORIGINAL/'summary.json'),
        audit=receipt(ROOT/'audit.json'),code=receipt(__file__))
    write(ROOT/'comparison.json',result);write('docs/experiments/2026-10-12-reference-groups-comparison.json',result)
    print(macro,flush=True);print(modifications,flush=True)


if __name__=='__main__':main()
