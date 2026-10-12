"""Complete edit-cost comparison with sealed composed and free controls."""
from itertools import combinations
from pathlib import Path
from scripts.research.run_query_field_reference_edit import ROOT,PREVIOUS,ARMS,design
from scripts.research.run_query_field_relations import read,write,receipt,check
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();audit=read(ROOT/'audit.json');assert audit['passed']
    for r in read(ROOT/'predictions_complete.json')['records']+read(ROOT/'scoring_complete.json')['records']:check(r)
    new=read(ROOT/'summary.json');old=read(PREVIOUS/'summary.json')
    grouped_path=Path(d['grouped_summary']['path']);grouped=read(grouped_path)
    arms=['composed','free','none','edit']
    per={'composed':old['per_case']['composed'],'free':grouped['per_case']['free'],**new['per_case']}
    macro={'composed':old['macro']['composed'],'free':grouped['macro']['free'],**new['macro']}
    comparisons={a+'__'+b:{k:paired({s:v[k] for s,v in per[a].items()},
        {s:v[k] for s,v in per[b].items()},'NVOS '+k) for k in macro[a]} for a,b in combinations(arms,2)}
    modifications={}
    for arm in ['free']+ARMS:
        summary=grouped if arm=='free' else new
        rows=[summary['diagnostics'][arm][j['name']]['diagnostics']['extent_decision'] for j in d['jobs']]
        modifications[arm]=dict(added_members=sum(r['added'] for r in rows),removed_members=sum(r['removed'] for r in rows),
            cases_with_any_change=sum(r['added']+r['removed']>0 for r in rows))
    reference={a:{k:sum(c['conditions'][a]['endpoints']['final'][k] for c in audit['cases'].values())/8
        for k in ['reference_set_iou','reference_observation_iou','edit_footprint']} for a in ARMS}
    result=dict(macro=macro,per_case=per,comparisons=comparisons,member_modifications=modifications,
        reference_agreement=reference,primary='edit',old_controls_rerun=False,
        objectives_not_directly_comparable=True,development_cases_not_heldout=True,
        raw_scribble=False,whole_five_task=False,default_promoted=False,
        summary=receipt(ROOT/'summary.json'),old_summary=receipt(PREVIOUS/'summary.json'),
        free_summary=receipt(grouped_path),audit=receipt(ROOT/'audit.json'),code=receipt(__file__))
    write(ROOT/'comparison.json',result);write('docs/experiments/2026-10-12-reference-edit-comparison.json',result)
    print(macro,flush=True);print(modifications,flush=True)


if __name__=='__main__':main()
