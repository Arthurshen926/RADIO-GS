"""Add immutable explicit query/recipe bindings to saved temporary states."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file
root=Path('output/nvos_reference_completion_next19')
r=json.loads((root/'execution.json').read_text());assert r['complete']
parameter_digest=sha256_file(root/'registration.json');records={}
for task in r['tasks']:
 report=json.loads((root/task/'report.json').read_text());receipt_path=Path(report['query_root'])/task/'receipt.json';qr=json.loads(receipt_path.read_text())
 for variant in r['variants']:
  state=root/task/(variant+'_elements.pt');side=Path(str(state)+'.binding.json');assert not side.exists()
  binding=dict(schema='nvos_temporary_query_state_binding.v1',stage=variant,scene=task,state_sha256=sha256_file(state),field_sha256=report['fields'][variant],geometry_sha256=report['geometry_sha256'],positive_sha256=qr['official_scribbles']['positive']['sha256'],sam_mask_sha256=qr['mask_sha256'],query_receipt_sha256=sha256_file(receipt_path),parameters_sha256=parameter_digest,source_code_sha256=sha256_file('scripts/audits/nvos_reference_completion_next19_evaluate.py'),semantics='post-component query state, not pre-component prompt_sam base')
  side.write_text(json.dumps(binding,indent=2)+'\n');records[str(side)]=sha256_file(side)
(root/'state_bindings.json').write_text(json.dumps(records,indent=2)+'\n');print(len(records))
