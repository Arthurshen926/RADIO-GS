"""Extend frozen singleton completion to monotonic native-support candidate union."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--preserve-precision',action='store_true');a=p.parse_args()
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'native_within_extent96';report=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==report['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False)
    architecture=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],identity_selection=dict(payload['architecture']['grounded_transfer']['identity_selection'],combine_candidates='greedy_union',union_preserve_precision=a.preserve_precision)))
    out=root/('native_precision_union96' if a.preserve_precision else 'native_support_union96');out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=architecture),out/'field.pt')
    r=dict(schema='native_support_union.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=report['field_sha256'],source_only=True,benchmark_data_used=False,all_parent_tensors_unchanged=True,
        policy='source-accepted extent unchanged; fallback greedily unions frozen candidates only while observed native-support IoU strictly improves; keep native ordering within the resulting membership strata',
        union_preserve_precision=a.preserve_precision,implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(out)


if __name__=='__main__':main()
