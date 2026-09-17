"""Scope image-supervised correction to image queries; preserve text/prompt."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--root',type=Path);p.add_argument('--retained',default='exact_render_query_threshold96');p.add_argument('--learned-parent',default='native_residual_rendered_selected');p.add_argument('--name',default='image_supervised_native_residual');a=p.parse_args();torch.set_num_threads(4)
root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
parent=root/a.learned_parent;r=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==r['field_sha256']
payload=torch.load(parent/'field.pt',weights_only=False)
arch=dict(payload['architecture'],native_semantic_residual=dict(payload['architecture']['native_semantic_residual'],modalities=['image']))
field=ElementRetrievalField(**arch);field.load_state_dict(payload['state_dict'],strict=True)
retained=root/a.retained;control=torch.load(retained/'field.pt',weights_only=False)
assert sha256_file(retained/'field.pt')==json.loads((retained/'report.json').read_text())['field_sha256']
assert payload['geometry_binding']==control['geometry_binding'] and payload['query_encoder']==control['query_encoder']
for k,v in control['state_dict'].items():assert torch.equal(v,field.state_dict()[k]),k
out=root/a.name;out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch),out/'field.pt')
report=dict(schema='image_supervised_native_residual.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=r['field_sha256'],retained_control_sha256=sha256_file(retained/'field.pt'),
 source_only_training=True,benchmark_data_used_for_fitting=False,development_informed_architecture=True,
 readout_policy='image-only native residual; original text and registered prompt paths retained exactly.',
 source_training_objective=r.get('supervision','source image retrieval'),source_training_report_sha256=sha256_file(parent/'report.json'),
 all_retained_parent_tensors_exact=True,no_additional_element_state=True,extra_global_parameters=sum(v.numel() for k,v in field.state_dict().items() if k.startswith('native_residual_decoder.')),
 source_validation_selected_step=r['selected_step'],relative_bound=arch['native_semantic_residual']['relative_bound'],
 status='development candidate; independent scene/benchmark acceptance required',
 implementation_sources={f:sha256_file(f) for f in [__file__,'radio_gs/models/element_retrieval_field.py','radio_gs/models/bounded_native_residual.py']})
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(a.scene,report['field_sha256'])
