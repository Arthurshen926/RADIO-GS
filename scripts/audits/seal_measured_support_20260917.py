"""Seal source-support ablations, current sources, and actual retention results."""
import json, shutil
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
out=Path('paper/artifacts/measured_support_seal_20260917.json')
snapshot=base/'measured_support_sources_20260917'
assert not out.exists()
files=['radio_gs/models/element_retrieval_field.py','radio_gs/models/bounded_native_residual.py',
 'radio_gs/models/native_decoder_readout.py','radio_gs/rendering/modality_support.py',
 'radio_gs/querying/signed_native_image_query.py','radio_gs/scripts/build_frozen_geometry_authority.py',
 'scripts/audits/build_native_image_support_extension.py','scripts/audits/export_measured_support_control.py',
 'scripts/audits/evaluate_nvos_frozen_image_route.py','scripts/audits/evaluate_real_reference_queries.py',
 'scripts/audits/evaluate_functional_queries.py','scripts/audits/verify_native_support_extension.py',
 'tests/test_image_support_extension.py','tests/test_modality_support.py','tests/test_signed_native_image_query.py','scripts/audits/seal_measured_support_20260917.py']
scenes={};artifacts={}
for scene in ['figurines','ramen']:
 p=Path(f'paper/artifacts/{scene}_image_support512_reference_20260917.json')
 d=json.loads(p.read_text());b=d['runs']['image_supervised_native_residual'];a=d['runs']['image_supervised_native_residual_measured_support512']
 for x,y in zip(b['observations'],a['observations']):
  assert (x['reference'],x['target_frame'],x['category'])==(y['reference'],y['target_frame'],y['category'])
  assert x['text']==y['text'] and x['prompt']==y['prompt']
 assert len(b['observations'])==len(a['observations'])
 old=json.loads(Path(f'paper/artifacts/{scene}_image_supervised_native_residual_reference_20260917.json').read_text())['runs']['image_supervised_native_residual']
 assert b['observations']==old['observations'], 'legacy evaluator changed'
 root=base/'preserved_source_inventory_20260916'/scene
 for name,r in d['runs'].items():assert sha256_file(root/name/'field.pt')==r['field_sha256']
 scenes[scene]={'before':b['macro'],'after':a['macro'],'text_prompt_observations_exact':True,'old_evaluator_observations_exact':True,
 'support_elements':a['support_elements'],'extension_report_sha256':sha256_file(root/'image_supervised_native_residual_measured_support512/report.json')}
 for suffix in ['', '.protocol', '.dispatch']:
  f=p if not suffix else p.with_suffix(suffix+'.json');artifacts[str(f)]=sha256_file(f)
 f=Path(f'paper/artifacts/{scene}_support512_retention_20260917.json');artifacts[str(f)]=sha256_file(f)
nvos={}
for variant in ['image_route','signed_image_route','measured_support','signed_measured_support','measured_support512','signed_measured_support512']:
 p=base/f'nvos_fern_{variant}_20260917/report.json';d=json.loads(p.read_text())
 assert not d['target_rgb_opened'] and d['target_gt_used_after_prediction_only']
 for name,digest in d['fields'].items():assert sha256_file(base/'nvos_frozen_authority_20260916/fern'/name/'field.pt')==digest
 nvos[variant]=d['results'];artifacts[str(p)]=sha256_file(p)
sources={}
for f in files:
 dest=snapshot/f;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert sha256_file(dest)==sha256_file(f)
 else:shutil.copy2(f,dest)
 sources[f]={'sha256':sha256_file(f),'snapshot':str(dest)}
out.write_text(json.dumps(dict(schema='measured_support_seal.v1',lerf=scenes,nvos_fern=nvos,artifacts=artifacts,sources=sources,
 verdict='Support extension improves fern but is essentially neutral on LERF; no joint promotion or independent final-test claim.',
 source_version_scope='Current snapshots include reusable geometry-bound retention check; historical evaluation receipts keep their executed source identities.'),indent=2)+'\n')
print(out)
