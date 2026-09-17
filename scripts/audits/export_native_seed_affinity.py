"""Export a fixed seed-to-extent readout with no feature or learned-weight change."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='scale_affinity_v1_direct_control');p.add_argument('--name',default='native_seed_affinity_v1');p.add_argument('--full-prompt',action='store_true');a=p.parse_args()
 root=a.root;parent=root/a.parent;report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256'];payload=torch.load(parent/'field.pt',weights_only=False)
 architecture=dict(payload['architecture']);architecture['scale_affinity']=dict(architecture['scale_affinity'],native_seed_queries=not a.full_prompt,full_prompt_features=a.full_prompt)
 out=root/a.name;out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=architecture),out/'field.pt')
 report=dict(report,field_sha256=sha256_file(out/'field.pt'),native_seed_readout=not a.full_prompt,native_text_image_max_error=0.0 if a.full_prompt else None,native_features_unchanged=True,prompt_response_unchanged=True,export_parent_sha256=sha256_file(parent/'field.pt'),export_implementation_sha256=sha256_file(__file__),query_contract='native cosine max/min selects one positive/negative seed; frozen structural affinity predicts extent; positive-query single-group diagnostic, absence and category union not guaranteed')
 if a.full_prompt:report.update(query_contract='direct registered signed readout from full 96D field; no gate or scale selection; native text/image unchanged',selection_policy='none; ungated full-vector control',prompt_response_unchanged=False)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
