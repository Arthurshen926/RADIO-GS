"""Run identical source-fusion/compression/readout diagnostics after each build."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);a=p.parse_args()
base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
for tag,folder in [('singlegrid','lerf_ropefix_pipeline_20260917'),('multiscale','lerf_ropefix_multiscale_pipeline_20260917'),('viewbalanced','lerf_view_balanced_rebuild_20260917')]:
 root=base/folder/a.scene;out=Path(f'paper/artifacts/{a.scene}_{tag}_ropefixed_native_compression_20260917.json')
 if out.exists():continue
 while True:
  try:r=json.loads((root/'execution.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
  if any(s['status']=='failed' for s in r['stages']):raise RuntimeError(str(root)+' failed; inspect before diagnostic')
  if r.get('finished'):break
  time.sleep(5)
 command=[sys.executable,'scripts/audits/diagnose_current_native_compression.py','--scene',a.scene,'--device',a.device,'--root',str(root/'field'),'--run','exact_render_query_threshold96','--output',str(out)]
 log=root/'native_compression.log'
 with log.open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(a.scene,tag,'native decomposition complete',flush=True)
