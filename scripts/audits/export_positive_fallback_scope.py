"""Export source-validated positive-only fallback without changing field tensors."""
import argparse,json,shutil
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 fit=a.root/'positive_fallback_source_control/report.json';r=json.loads(fit.read_text())
 assert r['source_acceptance'] and r['positive_only_diagnostic'] and r['source_concept_matches_withheld']
 parent=a.root/'joint_supervised_image_residual/field.pt';assert sha256_file(parent)==r['parent_field_sha256'];payload=torch.load(parent,weights_only=False)
 selection=dict(payload['architecture']['grounded_transfer']['identity_selection'],completion_mode='native_bounded_extent',scope_logit_bound=r['selected_strength'],preserve_native_positive=True)
 arch=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],identity_selection=selection))
 cold=ElementRetrievalField(**arch);cold.load_state_dict(payload['state_dict'],strict=True)
 out=a.root/'positive_fallback_scope'
 if out.exists():
  saved=torch.load(out/'field.pt',weights_only=False);assert saved['architecture']==arch
  assert all(torch.equal(v,saved['state_dict'][k]) for k,v in payload['state_dict'].items())
  print('Existing export exactly matches source-selected architecture and tensors');return
 out.mkdir();torch.save(dict(payload,architecture=arch),out/'field.pt');snap=out/'sources';snap.mkdir();impl={}
 for f in [__file__,'radio_gs/models/element_retrieval_field.py','radio_gs/models/bounded_scope_evidence.py']:
  shutil.copy2(f,snap/Path(f).name);impl[str(f)]=sha256_file(f)
 (out/'report.json').write_text(json.dumps(dict(source_only=True,benchmark_data_used=False,field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=sha256_file(parent),source_fit_sha256=sha256_file(fit),selected_strength=r['selected_strength'],all_parent_tensors_exact=True,implementations=impl),indent=2)+'\n')

if __name__=='__main__':main()
