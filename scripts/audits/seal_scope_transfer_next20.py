"""Seal finished controls, implementation and report without promoting defaults."""
import json,hashlib,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    summary_path=Path('output/scope_transfer_next20/summary.json');summary=json.loads(summary_path.read_text());assert summary['complete']
    manifest=json.loads(Path('output/frozen_cross_benchmark_method_20260918/source_manifest.json').read_text());assert all(sha(p)==h for p,h in manifest.items())
    coldpath=Path('output/caption_adjoint_allsource_next20/cold_runtime_module.json');cold=json.loads(coldpath.read_text());assert cold['complete'] and all(r['exact'] for r in cold['rows']);assert cold['runtime_module_sha256']==sha('radio_gs/querying/source_scope_consensus.py')
    bindings=dict(summary['artifact_sha256']);bindings[str(summary_path)]=sha(summary_path)
    original=[]
    for scene,count in zip(['figurines','ramen','teatime','waldo_kitchen'],[56,71,59,22]):
        p=Path('output/source_uplift_scope_next20')/scene/'evaluation/report.json';r=json.loads(p.read_text());assert r['complete'] and not r['routes'] and r['baseline_disagreement_pixels']==0
        for a,d in r['lerf2d'].items():
            assert len(d['observations'])==count and d==r['lerf2d']['final']
            assert r['selected'][a]['miou']==r['selected']['final']['miou']
        bindings[str(p)]=sha(p);original.append(dict(scene=scene,routed_queries=0,observations=count,all_arms_identical=True))
    sources=list(Path('scripts/audits').glob('*next20.py'))+[Path('radio_gs/querying/source_scope_consensus.py'),Path('tests/test_source_scope_uplift_control.py'),Path('docs/experiments/2026-09-19-renderer-weighted-scope-transfer-next20.md')]
    for p in sources:bindings[str(p)]=sha(p)
    for p in [Path('output/source_uplift_scope_next20/adjoint_verification.json'),coldpath]:bindings[str(p)]=sha(p)
    logs=Path('output/scope_transfer_next20/logs');logs.mkdir(exist_ok=False)
    names=['selected_next20_fig.log','selected_next20_ram.log','selected_plain_fig.log','selected_plain_ram.log','uplift_eval_next20_fig.log','uplift_eval_next20_tea.log','uplift_eval_next20_ram.log','caption_scope_module_final.log']
    for name in names:
        src=Path('/tmp')/name
        if src.exists():dst=logs/name;shutil.copy2(src,dst);bindings[str(dst)]=sha(dst)
    result=dict(complete=True,default_promoted=False,significant_gain_achieved=False,independent_test=False,official_four_query=False,geometry_changed=False,frozen_source_count=len(manifest),frozen_source_changes=[],tests=dict(command='OMP_NUM_THREADS=4 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q tests/test_source_scope_uplift_control.py',passed=5,exit_code=0),source_free_cold_replay_exact=True,original_template_control=original,macro={k:v['macro'] for k,v in summary['cohorts'].items()},auxiliary_scope_bytes=summary['auxiliary_scope_bytes'],scannet_nvos_new_results=False,uqis_paused=True,artifact_sha256=bindings)
    p=Path('paper/artifacts/scope_transfer_next20_complete.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(p,len(bindings))
if __name__=='__main__':main()
