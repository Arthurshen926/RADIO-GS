"""Use native element support IoU to select unchanged learned extents."""
import argparse,json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--fallback-only',action='store_true');p.add_argument('--root',type=Path);a=p.parse_args()
    torch.set_num_threads(4)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parent=root/'source_hierarchical_candidates96';r=json.loads((parent/'report.json').read_text())
    assert sha256_file(parent/'field.pt')==r['field_sha256']
    payload=torch.load(parent/'field.pt',weights_only=False)
    for suffix,restrict in ([('fallback',False)] if a.fallback_only else [('restricted',True),('global',False)]):
        policy=dict(metric='support_iou',restrict_concepts=restrict,require_native_support=False)
        if a.fallback_only:policy['trusted_concept_start']=r['independent_region_count']
        arch=dict(payload['architecture'],grounded_transfer=dict(payload['architecture']['grounded_transfer'],identity_selection=policy))
        field=ElementRetrievalField(**arch);missing=field.load_state_dict(payload['state_dict'],strict=False)
        assert missing.missing_keys==['grounded_transfer.native_extent_means'] and not missing.unexpected_keys
        state=field.state_dict()
        for k,v in payload['state_dict'].items():assert torch.equal(v,state[k]),k
        out=root/('native_overlap_'+suffix+'96');out.mkdir(exist_ok=False)
        torch.save(dict(payload,architecture=arch,state_dict=state),out/'field.pt')
        report=dict(schema='native_support_overlap.v1',field_sha256=sha256_file(out/'field.pt'),parent_field_sha256=r['field_sha256'],
            source_only=True,benchmark_data_used=False,all_parent_tensors_unchanged=True,identity_policy=policy,
            support_contract='observed elements with native cosine above unchanged source query threshold; candidate raw membership > 0.5; full element IoU; zero overlap abstains',
            implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(out,flush=True)


if __name__=='__main__':main()
