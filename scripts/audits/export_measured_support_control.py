"""Matched image support extension without the learned native residual."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',required=True);p.add_argument('--extension',default='image_supervised_native_residual_measured_support');p.add_argument('--output-name');a=p.parse_args();torch.set_num_threads(4)
root=a.root;path=root/a.extension/'field.pt';ep=torch.load(path,weights_only=False)
assert sha256_file(path)==json.loads((path.parent/'report.json').read_text())['field_sha256']
base=root/a.base/'field.pt';payload=torch.load(base,weights_only=False);assert sha256_file(base)==json.loads((base.parent/'report.json').read_text())['field_sha256']
assert payload['geometry_binding']==ep['geometry_binding'] and payload['query_encoder']==ep['query_encoder']
state=dict(payload['state_dict'],latent=ep['state_dict']['latent'],image_observed=ep['state_dict']['image_observed']);arch=dict(payload['architecture'],image_support_extension=True)
f=ElementRetrievalField(**arch);f.load_state_dict(state,strict=True)
for k,v in payload['state_dict'].items():
 if k=='latent':assert torch.equal(v[f.observed],state[k][f.observed])
 else:assert torch.equal(v,state[k])
out=root/(a.output_name or a.base+'_measured_support');out.mkdir(exist_ok=False);torch.save(dict(payload,state_dict=state,architecture=arch),out/'field.pt')
(out/'report.json').write_text(json.dumps(dict(schema='matched_measured_support_control.v1',field_sha256=sha256_file(out/'field.pt'),original_field_sha256=sha256_file(base),extension_source_field_sha256=sha256_file(path),same_measured_extension=True,source_only=True,benchmark_data_used=False,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')
