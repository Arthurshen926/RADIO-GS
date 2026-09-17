"""Check all source calibration phrases, not just a single cold template."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
from radio_gs.models.functional_query_encoder import FunctionalThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);assert not a.output.exists()
root=Path('output/lerf_ropefix_multiscale_pipeline_20260917')/a.scene/'field';fp=root/'source_precision_gate96/field.pt';r=json.loads(fp.with_name('report.json').read_text());assert sha256_file(fp)==r['field_sha256'];b=torch.load(fp,map_location='cpu',weights_only=False);arch=b['architecture'];h=GroundedSemanticTransfer(arch['latent_dim'],arch['native_semantic_dim'],arch['query_dims']['text'],**arch['grounded_transfer']).to(a.device);h.load_state_dict({k[len('grounded_transfer.'):]:v for k,v in b['state_dict'].items() if k.startswith('grounded_transfer.')});h.requires_grad_(False)
pth=root/'episodes.pt';assert sha256_file(pth)==r['episode_sha256'];ep=torch.load(pth,map_location='cpu',weights_only=False);phrases=r['phrases'];encoder=FunctionalThreeQueryEncoder(b['query_encoder'],a.device)
with torch.no_grad():
 batch=encoder.encode_text(phrases);single=torch.cat([encoder.encode_text([s]) for s in phrases]);cached=batch.half().float().clone()
 for i,phrase in enumerate(ep['queries']['queries']):cached[phrases.index(phrase)]=ep['queries']['text'][i].to(a.device).float()
 variants={'batch_fp32':batch,'single_fp32':single,'batch_fp16':batch.half().float(),'single_fp16':single.half().float(),'cached_when_available':cached}
 results={};new=h.acceptance.clone()
 for label,threshold in [('original',torch.tensor([x['old_threshold'] for x in r['checks']],device=a.device)),('precision_repaired',new)]:
  h.acceptance.copy_(threshold);routes={}
  for name,q in variants.items():
   index,accepted,score=h.route(q);routes[name]=dict(index=index.tolist(),accepted=accepted.tolist())
  changes=[]
  for i,phrase in enumerate(phrases):
   decisions={(v['index'][i],v['accepted'][i]) for v in routes.values()}
   if len(decisions)>1:changes.append(dict(phrase=phrase,decisions=sorted(decisions)))
  full=h.route(batch.half().float());parts=[h.route(q[None]) for q in batch.half().float()];same_tensor_batch_changes=sum(not (int(full[0][i])==int(x[0][0]) and bool(full[1][i])==bool(x[1][0])) for i,x in enumerate(parts))
  results[label]=dict(encoding_variant_route_disagreements=len(changes),disagreements=changes,same_tensor_batch_route_changes=same_tensor_batch_changes)
report=dict(schema='source_gate_stability.v1',source_only=True,benchmark_data_used=False,field_sha256=r['field_sha256'],calibration_report_sha256=sha256_file(fp.with_name('report.json')),episode_sha256=r['episode_sha256'],source_phrases=len(phrases),variants=list(variants),results=results,future_query_stability_guaranteed=False,implementation_sha256=sha256_file(__file__))
a.output.write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,a.output.with_suffix('.source.py'));print(report,flush=True)
