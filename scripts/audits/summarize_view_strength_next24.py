"""Complete paired four-scene source-strength ablation; no per-scene cherry-picking."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

SCENES={'figurines':56,'ramen':71,'teatime':59,'waldo_kitchen':22}
ARMS=['final','lifted_calibrated','local_calibrated']


def main():
    bindings={};scenes={};pairs=[]
    def read(path):
        path=Path(path);bindings[str(path)]=sha256_file(path)
        return json.loads(path.read_text())
    for scene,count in SCENES.items():
        root=Path('output/caption_view_strength_next24')/scene
        current=read(root/'evaluation/report.json')
        source=read(root/'report.json')
        old=read(Path('output/caption_adjoint_allsource_next20')/scene/'evaluation/report.json')
        failed=read(Path('output/caption_view_consistency_next22')/scene/'evaluation/report.json')
        assert current['complete'] and current['baseline_disagreement_pixels']==0
        assert current['source_report_sha256']==bindings[str(root/'report.json')]
        assert sha256_file(root/'uplift.pt')==source['uplift_sha256']
        assert sha256_file(root/'evaluation/scores.pt')==current['scores_sha256']
        bindings[str(root/'uplift.pt')]=source['uplift_sha256']
        bindings[str(root/'evaluation/scores.pt')]=current['scores_sha256']
        assert current['lerf2d']['final']==old['lerf2d']['final']==failed['lerf2d']['final']
        for arm in ARMS:
            assert len(current['lerf2d'][arm]['observations'])==count
            assert current['selected'][arm]['count']==count and current['selected'][arm]['missing']==0
        metrics={a:{'iou':current['lerf2d'][a]['miou'],'ap':current['lerf2d'][a]['ap'],'selected':current['selected'][a]['miou']} for a in ARMS}
        metrics['previous_local']={'iou':old['lerf2d']['local_calibrated']['miou'],'ap':old['lerf2d']['local_calibrated']['ap'],'selected':old['selected']['local_calibrated']['miou']}
        metrics['previous_bad_one_sided']={'iou':failed['lerf2d']['local_calibrated']['miou'],'ap':failed['lerf2d']['local_calibrated']['ap'],'selected':failed['selected']['local_calibrated']['miou']}
        for arm in ['lifted_calibrated','local_calibrated']:
            ref={(v['frame'],v['query']):v for v in old['lerf2d']['local_calibrated']['observations']}
            for v in current['lerf2d'][arm]['observations']:
                pairs.append(dict(scene=scene,arm=arm,frame=v['frame'],query=v['query'],iou_delta_from_previous_local=v['iou']-ref[(v['frame'],v['query'])]['iou']))
        scenes[scene]=dict(metrics=metrics,source_in_sample=source['in_sample_nonregression'],source_accepted=source['source_accepted'],coverage=source['covered_element_query_fraction'],reliable=source['reliable_element_query_fraction'],replay_error=source['baseline_uplift_replay_max_error'])
    macro={arm:{key:sum(s['metrics'][arm][key] for s in scenes.values())/4 for key in ['iou','ap','selected']} for arm in scenes['figurines']['metrics']}
    manifest=read('output/frozen_cross_benchmark_method_20260918/source_manifest.json')
    changed=[p for p,h in manifest.items() if sha256_file(p)!=h];assert not changed
    files=['radio_gs/training/one_sided_scope_evidence.py','tests/test_one_sided_scope_evidence.py','scripts/audits/caption_view_strength_next24.py','scripts/audits/evaluate_caption_view_strength_next24.py',__file__]
    for p in files:bindings[p]=sha256_file(p)
    result=dict(complete=True,scenes=scenes,macro=macro,paired=pairs,observations_per_arm=208,frozen_source_count=len(manifest),frozen_source_changes=changed,default_promoted=False,independent_final_test=False,official_four_query=False,selected_is_volumetric_3d_iou=False,artifact_sha256=bindings)
    out=Path('output/view_strength_next24');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(macro,indent=2))

if __name__=='__main__':main()
