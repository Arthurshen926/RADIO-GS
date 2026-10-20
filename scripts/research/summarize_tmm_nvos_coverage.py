"""Build the fixed NVOS factorial table from complete verified reports only."""
import csv
import json
from pathlib import Path
from statistics import mean

from radio_gs.response_study.reference_study import (
    check_reference_study, check_coverage_ranking, check_inherited_readouts, read, sha,
)

REPO=Path(__file__).resolve().parents[2]
PAPER=REPO/'paper/tmm_response'


def main():
    registry=read(REPO/'configs/research/tmm_response.json')
    item=next(r for r in registry['reference_studies'] if r['id']=='N01')
    study=REPO/registry['study_root']
    jobs=read(REPO/registry['recipe'])['jobs']
    record=check_reference_study(REPO,study,item,jobs)
    if not record['complete']:
        raise ValueError('All eight NVOS tasks must complete before table generation')
    summary=read(study/item['run']/'summary.json')
    # This pass occurs only after all predictions have been sealed and scored.
    # It binds GT files and independently checks each recorded GT denominator.
    import numpy as np
    from PIL import Image
    gt_receipts={}
    for job in jobs:
        if job['benchmark']!='nvos':continue
        spec=read(REPO/job['evaluation']);path=Path(spec['gt_mask'])
        gt=np.asarray(Image.open(path));name=job['name']
        counts=dict(retained_gt_gt127=int((gt>127).sum()),
                    ludvig_gt_maximum=int((gt//gt.max()).astype(bool).sum()))
        for arm in summary['rows'][name].values():
            for track,expected in counts.items():
                if arm[track]['tp']+arm[track]['fn']!=expected:
                    raise ValueError('GT denominator differs from sealed scores: '+name)
        record['receipts'][str(path)]=sha(path)
        gt_receipts[name]=dict(path=str(path),sha256=sha(path),foreground_counts=counts)
    arms=['legacy_low','coverage_low','legacy_high','coverage_high']
    rows=summary['rows'];effects={};source_stats={};flat=[]
    for name,values in rows.items():
        report=read(study/item['run']/(name+'_prediction.json'))
        source_stats[name]={}
        for arm in arms:
            stats=report['arms'][arm]['source_evidence_statistics']
            total=sum(v['hard_foreground_pixels'] for v in stats)
            kept=sum(v['legacy_positive_pixels'] for v in stats)
            source_stats[name][arm]=dict(hard_foreground_pixels=total,eroded_positive_pixels=kept,
                positive_pixel_fraction_erased=1-kept/max(total,1))
            for track,metrics in values[arm].items():
                flat.append(dict(task=name,arm=arm,gt_track=track,**metrics))
    for track in summary['macro']:
        base={n:r['legacy_low'][track]['iou'] for n,r in rows.items()}
        effects[track]={}
        for arm in arms[1:]:
            delta={n:100*(r[arm][track]['iou']-base[n]) for n,r in rows.items()}
            clusters={n:('horns' if n.startswith('horns_') else n) for n in rows}
            unique=sorted(set(clusters.values()));rng=np.random.default_rng(0);boot=[]
            for sampled in rng.choice(unique,size=(5000,len(unique)),replace=True):
                values=[d for cluster in sampled for n,d in delta.items() if clusters[n]==cluster]
                boot.append(mean(values))
            effects[track][arm]=dict(delta_pp=mean(delta.values()),per_task_delta_pp=delta,
                improved=sum(x>1e-6 for x in delta.values()),harmed=sum(x< -1e-6 for x in delta.values()),
                descriptive_scene_bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist(),
                independent_scene_clusters=7,bootstrap_samples=5000)
        m=summary['macro'][track]
        effects[track]['factorial']=dict(
            evidence_at_low_pp=100*(m['coverage_low']-m['legacy_low']),
            evidence_at_high_pp=100*(m['coverage_high']-m['legacy_high']),
            resolution_at_legacy_pp=100*(m['legacy_high']-m['legacy_low']),
            resolution_at_coverage_pp=100*(m['coverage_high']-m['coverage_low']),
            interaction_pp=100*(m['coverage_high']-m['legacy_high']-m['coverage_low']+m['legacy_low']))
    record.update(effects=effects,source_erosion=source_stats,rows=rows,scoring_gt_receipts=gt_receipts,
                  hypothesis='low-resolution erosion loses useful reference evidence',
                  evidence_intervention='area/confidence/authority change together; not an isolated confidence-weight ablation',
                  resolution_intervention='source raster and geometric contribution sampling change; fixed geometry parameters',
                  no_new_all_five_inference=True,default_changed=False)
    if (study/'nvos_coverage_ranking_v2/summary.json').exists():
        ranking=check_coverage_ranking(study)
        record['ranking']=ranking['macro'];record['receipts'].update(ranking['receipts'])
    if (study/'nvos_v10_readout_audit_v1/summary.json').exists():
        inherited=check_inherited_readouts(study)
        record['inherited_readouts']=inherited;record['receipts'].update(inherited['receipts'])
        inherited_lines=[r'\begin{tabular}{lrrr}',r'\toprule',
            r'Task & v10 response & v10 independent set & $\Delta$ (pp) \\',r'\midrule']
        for name,result in inherited['rows'].items():
            a=result['semantic_response']['retained_gt_gt127']['iou']
            b=result['independent_set_projection']['retained_gt_gt127']['iou']
            inherited_lines.append(name.replace('_',r'\_')+f' & {100*a:.2f} & {100*b:.2f} & {100*(b-a):+.2f}'+r' \\')
        for track,label in [('retained_gt_gt127','Mean (retained GT)'),('ludvig_gt_maximum','Mean (LUDVIG GT)')]:
            values=inherited['macro'][track];a=values['semantic_response']['iou'];b=values['independent_set_projection']['iou']
            inherited_lines.append(r'\midrule '+label+f' & {100*a:.2f} & {100*b:.2f} & {100*(b-a):+.2f}'+r' \\')
        inherited_lines += [r'\bottomrule',r'\end{tabular}']
        (PAPER/'tables/nvos_inherited_readouts.tex').write_text('\n'.join(inherited_lines)+'\n')
    if (study/'nvos_v10_readout_factorial_v1/summary.json').exists():
        factorial=check_inherited_readouts(study,'nvos_v10_readout_factorial_v1')
        record['readout_factorial']=factorial;record['receipts'].update(factorial['receipts'])
        lines=[r'\begin{tabular}{llrrrr}',r'\toprule',
               r'Field & Readout & IoU & AP & Precision & Recall \\',r'\midrule']
        for arm,field,output in [('semantic_response','Semantic','Full scene'),
                                 ('semantic_set_projection','Semantic','Selected alpha'),
                                 ('independent_response','Member','Full scene'),
                                 ('independent_set_projection','Member','Selected alpha')]:
            values=factorial['macro']['retained_gt_gt127'][arm]
            lines.append(field+' & '+output+' & '+' & '.join(f'{100*values[k]:.2f}'
                for k in ('iou','average_precision','precision','recall'))+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (PAPER/'tables/nvos_readout_factorial.tex').write_text('\n'.join(lines)+'\n')
    (PAPER/'nvos_coverage_feedback.json').write_text(json.dumps(record,indent=2)+'\n')
    with (PAPER/'artifacts/nvos_coverage.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Task & Old / low & Coverage / low & Old / high & Coverage / high \\',r'\midrule']
    for name,values in rows.items():
        lines.append(name.replace('_',r'\_')+' & '+' & '.join(f"{100*values[a]['retained_gt_gt127']['iou']:.2f}" for a in arms)+r' \\')
    lines.append(r'\midrule')
    for track,label in [('retained_gt_gt127','Mean (retained GT)'),('ludvig_gt_maximum','Mean (LUDVIG GT)')]:
        lines.append(label+' & '+' & '.join(f"{100*summary['macro'][track][a]:.2f}" for a in arms)+r' \\')
    if 'ranking' in record:
        for track,label in [('retained_gt_gt127','AP (retained GT)'),('ludvig_gt_maximum','AP (LUDVIG GT)')]:
            lines.append(label+' & '+' & '.join(f"{100*record['ranking'][track][a]['average_precision']:.2f}" for a in arms)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (PAPER/'tables/nvos_coverage.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(macro=summary['macro'],effects=effects),indent=2))


if __name__=='__main__':main()
