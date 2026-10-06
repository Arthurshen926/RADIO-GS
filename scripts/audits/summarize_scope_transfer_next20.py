"""Pair complete cohorts; never choose per-scene benchmark winners."""
import json,hashlib
from pathlib import Path
import numpy as np
SCENES=['figurines','ramen','teatime','waldo_kitchen']
COUNTS=[56,71,59,22]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=Path('output/scope_transfer_next20');out.mkdir(exist_ok=False)
    cohorts={};bindings={};pairs=[]
    for folder in ['source_uplift_scope_plain_next20','caption_adjoint_allsource_next20','caption_adjoint_graph_next20']:
        reports={s:json.loads((Path('output')/folder/s/'evaluation/report.json').read_text()) for s in SCENES}
        for scene,expected in zip(SCENES,COUNTS):
            r=reports[scene];assert r['complete'] and r['baseline_disagreement_pixels']==0
            p=Path('output')/folder/scene/'evaluation/report.json';bindings[str(p)]=sha(p)
            scores=p.parent/'scores.pt';assert sha(scores)==r['scores_sha256'];bindings[str(scores)]=r['scores_sha256']
            baseline={(x['frame'],x['query']):x for x in r['lerf2d']['final']['observations']}
            baseline3={(x['frame'],x['query']):x for x in r['selected']['final']['objects']}
            for arm,d in r['lerf2d'].items():
                assert len(d['observations'])==expected and r['selected'][arm]['count']==expected and r['selected'][arm]['missing']==0
                actual3={(x['frame'],x['query']):x for x in r['selected'][arm]['objects']}
                for x in d['observations']:
                    key=(x['frame'],x['query']);pairs.append(dict(cohort=folder,scene=scene,arm=arm,frame=x['frame'],query=x['query'],iou_delta=x['iou']-baseline[key]['iou'],ap_delta=x['ap']-baseline[key]['ap'],selected_delta=actual3[key]['iou']-baseline3[key]['iou']))
        arms=list(reports[SCENES[0]]['lerf2d']);cohorts[folder]=dict(scenes={s:dict(lerf2d={a:{k:v for k,v in reports[s]['lerf2d'][a].items() if k!='observations'} for a in arms},selected={a:reports[s]['selected'][a]['miou'] for a in arms},routed_queries=len(reports[s]['routes'])) for s in SCENES},macro={a:dict(iou=float(np.mean([reports[s]['lerf2d'][a]['miou'] for s in SCENES])),ap=float(np.mean([reports[s]['lerf2d'][a]['ap'] for s in SCENES])),selected_iou=float(np.mean([reports[s]['selected'][a]['miou'] for s in SCENES]))) for a in arms})
    manifest=json.loads(Path('output/frozen_cross_benchmark_method_20260918/source_manifest.json').read_text());changed=[p for p,h in manifest.items() if sha(p)!=h];assert not changed
    cold=Path('output/caption_adjoint_allsource_next20/cold_runtime_module.json');c=json.loads(cold.read_text());assert c['complete'];assert c['runtime_module_sha256']==sha('radio_gs/querying/source_scope_consensus.py');bindings[str(cold)]=sha(cold)
    for folder in ['source_selected_calibration_next20','source_selected_plain_calibration_next20','source_uplift_scope_next20','source_uplift_scope_plain_next20','caption_adjoint_scope_next20','caption_adjoint_allsource_next20']:
        for p in sorted((Path('output')/folder).glob('*/report.json')):
            d=json.loads(p.read_text());assert d['complete'];bindings[str(p)]=sha(p)
            if 'uplift_sha256' in d:
                cache=p.parent/'uplift.pt';assert sha(cache)==d['uplift_sha256'];bindings[str(cache)]=d['uplift_sha256']
    result=dict(complete=True,scenes=SCENES,observations_per_arm=208,cohorts=cohorts,paired_observations=pairs,independent_test=False,official_four_query=False,default_promoted=False,geometry_unchanged=True,frozen_source_count=len(manifest),frozen_source_changes=changed,auxiliary_scope_bytes=sum(x['uplift_bytes'] for x in c['rows']),cold_source_free_exact=True,artifact_sha256=bindings)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['macro'] for k,v in cohorts.items()},indent=2))

if __name__=='__main__':main()
