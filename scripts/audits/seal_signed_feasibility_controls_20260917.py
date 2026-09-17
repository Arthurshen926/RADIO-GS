"""Seal paired signed-query controls, preserving their legacy-SAM limitation."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
base=Path('/mnt/pool/sqy/results/RADIO-GS/output');out=Path('paper/artifacts/signed_feasibility_controls_seal_20260917.json');assert not out.exists();artifacts={};sources={};metrics={}
for scene,policy in [('fern','adaptive'),('fortress','adaptive'),('flower','adaptive'),('orchids','adaptive'),('fern','minimal'),('fortress','minimal'),('orchids','stable_fixedcap')]:
 root=base/f'nvos_{scene}_{policy}_signed_20260917';p=root/'report.json';r=json.loads(p.read_text());artifacts[str(p)]=sha256_file(p)
 assert not r['target_rgb_opened'] and r['target_gt_used_after_prediction_only']
 for name,result in r['results'].items():
  p=root/(name+'.pt');assert sha256_file(p)==result['prediction_sha256'];artifacts[str(p)]=result['prediction_sha256']
 for f,h in r['implementation_sources'].items():
  p=root/'sources'/Path(f).name;assert sha256_file(p)==h;sources[str(p)]=h
 metrics[scene+'_'+policy]=r['results']
out.write_text(json.dumps(dict(schema='signed_feasibility_controls_seal.v1',artifacts=artifacts,sources=sources,metrics=metrics,source_and_query_sam_lineage='legacy complex-RoPE-corrupted point-only inference',causal_claim='Same stable solver and budget fixed-cap control isolates signed correction feasibility on orchids.',clean_teacher_architecture_acceptance=False,independent_final_test=False),indent=2)+'\n');print(out)
