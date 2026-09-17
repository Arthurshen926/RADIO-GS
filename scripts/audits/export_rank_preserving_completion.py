"""Make structural completion calibrate a native cutoff, never replace ranking."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--within-extent',action='store_true');p.add_argument('--root',type=Path);a=p.parse_args()
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'hard_source_scope_gate96';r=json.loads((parent/'report.json').read_text());assert sha256_file(parent/'field.pt')==r['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False)
    mode='native_within_extent' if a.within_extent else 'native_threshold'
    arch=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],identity_selection=dict(payload['architecture']['grounded_transfer']['identity_selection'],completion_mode=mode)))
    out=root/('native_within_extent96' if a.within_extent else 'rank_preserving_completion96');out.mkdir(exist_ok=False);torch.save(dict(payload,architecture=arch),out/'field.pt')
    (out/'report.json').write_text(json.dumps(dict(schema='rank_preserving_completion.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=r['field_sha256'],source_only=True,benchmark_data_used=False,all_parent_tensors_unchanged=True,
        policy=('source-accepted extents unchanged; fallback probability = 0.5 * (structural binary membership + native probability); native ranks preserved within membership strata, not across them' if a.within_extent else 'source-accepted extents unchanged; structural fallback determines native-score cutoff maximizing observed-element IoU with selected extent; ties cannot split; no benchmark threshold search'),
        implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/element_retrieval_field.py')),indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()
