"""Remove only the concept extent readout for a paired native-space diagnostic."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--output-name');a=p.parse_args();torch.set_num_threads(4)
 path=a.root/a.parent/'field.pt';report=json.loads(path.with_name('report.json').read_text());assert sha256_file(path)==report['field_sha256'];payload=torch.load(path,weights_only=False)
 if payload['architecture'].get('bounded_extent'):raise ValueError('bounded extent requires its concept head; cannot silently remove it')
 arch=dict(payload['architecture'],grounded_transfer=None);state={k:v for k,v in payload['state_dict'].items() if not k.startswith('grounded_transfer.')}
 field=ElementRetrievalField(**arch);field.load_state_dict(state,strict=True)
 out=a.root/(a.output_name or a.parent+'_native_fallback_control');out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch,state_dict=state),out/'field.pt')
 (out/'report.json').write_text(json.dumps(dict(schema='native_fallback_control.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=report['field_sha256'],change='remove grounded_transfer branch only; preserve every other tensor',source_only=True,benchmark_data_used=False,implementation_sha256=sha256_file(__file__)),indent=2)+'\n')

if __name__=='__main__':main()
