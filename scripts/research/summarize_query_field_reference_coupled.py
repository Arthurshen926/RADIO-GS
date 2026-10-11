"""Compare all fixed arms while keeping sealed prior controls identifiable."""
from itertools import combinations
from scripts.research.run_query_field_reference_coupled import ROOT,PREVIOUS,design
from scripts.research.run_query_field_relations import read,write,receipt,check
from scripts.research.summarize_query_field_relations import paired


def main():
    d=design();audit=read(ROOT/'audit.json');assert audit['passed']
    old=read(PREVIOUS/'summary.json');new=read(ROOT/'summary.json')
    arms=['composed','shared_prior','coupled','observation']
    macro={a:(old if a in old['macro'] else new)['macro'][a] for a in arms}
    per={a:(old if a in old['per_case'] else new)['per_case'][a] for a in arms}
    for j in d['jobs']:
        old_cfg=read(PREVIOUS/'configs'/(j['name']+'.json'));cfg=read(j['configuration']['path'])
        base={k:v for k,v in old_cfg['jobs'][0].items() if k not in {'name','query_program'}}
        assert all({k:v for k,v in job.items() if k not in {'name','query_program'}}==base for job in cfg['jobs'])
        for k in ['projection_hw','output_hw','alpha_threshold']:assert cfg[k]==old_cfg[k]
    comparisons={a+'__'+b:{k:paired({s:v[k] for s,v in per[a].items()},
        {s:v[k] for s,v in per[b].items()},'NVOS '+k) for k in macro[a]} for a,b in combinations(arms,2)}
    reference={a:{k:sum(c['conditions'][a]['endpoints']['final'][k] for c in audit['cases'].values())/8
        for k in ['reference_set_iou','reference_observation_iou']} for a in ['coupled','observation']}
    result=dict(macro=macro,per_case=per,comparisons=comparisons,current_reference_fit=reference,
        primary='coupled',same_current_inputs_checked=True,old_controls_rerun=False,
        development_cases_not_heldout=True,whole_five_task=False,default_promoted=False,
        previous_summary=receipt(PREVIOUS/'summary.json'),new_summary=receipt(ROOT/'summary.json'),
        audit=receipt(ROOT/'audit.json'),code=receipt(__file__))
    write(ROOT/'comparison.json',result)
    write('docs/experiments/2026-10-12-reference-coupled-comparison.json',result)
    print(macro,flush=True)


if __name__=='__main__':main()
