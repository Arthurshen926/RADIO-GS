"""Seal complete fixed-cohort controls, including unsuccessful method changes."""
import json
from collections import Counter
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    scenes=dict(figurines=56,ramen=71,teatime=59,waldo_kitchen=22)
    bindings={}
    def read(p):
        p=Path(p);bindings[str(p)]=sha256_file(p);return json.loads(p.read_text())
    rows={};pairs=[]
    for s,n in scenes.items():
        root=Path('output/caption_view_consistency_next22')/s
        r=read(root/'evaluation/report.json');old=read(Path('output/caption_adjoint_allsource_next20')/s/'evaluation/report.json')
        source=read(root/'report.json');relations=read(root/'view_relations.json')
        assert r['complete'] and r['baseline_disagreement_pixels']==0
        assert sha256_file(root/'uplift.pt')==source['uplift_sha256']
        assert sha256_file(root/'evaluation/scores.pt')==r['scores_sha256']
        bindings[str(root/'uplift.pt')]=source['uplift_sha256']
        bindings[str(root/'evaluation/scores.pt')]=r['scores_sha256']
        assert r['source_report_sha256']==bindings[str(root/'report.json')]
        for arm,d in r['lerf2d'].items():
            assert len(d['observations'])==n and r['selected'][arm]['count']==n and r['selected'][arm]['missing']==0
        assert r['lerf2d']['final']==old['lerf2d']['final']
        arms={a:dict(iou=v['miou'],ap=v['ap'],selected_iou=r['selected'][a]['miou']) for a,v in r['lerf2d'].items()}
        arms['previous_local']=dict(iou=old['lerf2d']['local_calibrated']['miou'],ap=old['lerf2d']['local_calibrated']['ap'],selected_iou=old['selected']['local_calibrated']['miou'])
        oldby={(x['frame'],x['query']):x for x in old['lerf2d']['local_calibrated']['observations']}
        for x in r['lerf2d']['local_calibrated']['observations']:
            pairs.append(dict(scene=s,frame=x['frame'],query=x['query'],iou_delta_from_previous=x['iou']-oldby[(x['frame'],x['query'])]['iou']))
        rows[s]=dict(arms=arms,relations=dict(Counter(x['relation'] for x in relations['observations'])),reliable_fraction=source['reliable_element_query_fraction'],covered_fraction=source['covered_element_query_fraction'],source_replay_error=source['baseline_uplift_replay_max_error'],source_in_sample=source['in_sample_nonregression'])
    macro={a:{k:sum(v['arms'][a][k] for v in rows.values())/4 for k in ['iou','ap','selected_iou']} for a in rows['figurines']['arms']}
    nv=read('output/nvos_bridge_next22/execution.json');assert nv['complete'] and len(nv['stages'])==8 and all(x['returncode']==0 for x in nv['stages'])
    nrows={}
    for s in nv['tasks']:
        root=Path('output/nvos_bridge_next22')/s;r=read(root/'report.json')
        assert r['baseline_mask_replay_exact'] and r['baseline_response_max_abs_error']<=1e-6
        assert not r['target_rgb_opened'] and not r['negative_scribble_opened']
        assert len(set(r['fields'].values()))==1
        for a,v in r['results'].items():assert sha256_file(root/(a+'.pt'))==v['prediction_sha256']
        nrows[s]=dict(results={a:{k:v[k] for k in ['iou','average_precision','pixel_accuracy']} for a,v in r['results'].items()},baseline_max_error=r['baseline_response_max_abs_error'],diagnostics={a:{k:v[k] for k in ['uncertain_connectors','retained_elements','filter_selected_elements','reintroduced_elements','reference_after_filter']} for a,v in r['refinement_records'].items()})
    manifest=read('output/frozen_cross_benchmark_method_20260918/source_manifest.json')
    changed=[p for p,h in manifest.items() if sha256_file(p)!=h];assert not changed
    sources=['radio_gs/training/scope_view_relation.py','radio_gs/querying/uncertain_scope_bridge.py','tests/test_scope_view_relation.py','tests/test_uncertain_scope_bridge.py']+[str(p) for p in Path('scripts/audits').glob('*next22.py')]
    for p in sources:bindings[p]=sha256_file(p)
    result=dict(complete=True,lerf=dict(observations_per_arm=208,scenes=rows,macro=macro,paired=sorted(pairs,key=lambda x:x['iou_delta_from_previous'])),nvos=dict(tasks=nrows,macro=nv['macro']),frozen_sources=len(manifest),frozen_source_changes=changed,default_promoted=False,independent_final_test=False,official_four_query=False,selected_metric_is_volumetric_3d_iou=False,scannet_new_run=False,uqis_paused=True,artifact_sha256=bindings)
    out=Path('output/scope_connectivity_next22');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(lerf=macro,nvos=nv['macro']),indent=2))

if __name__=='__main__':main()
