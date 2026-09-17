"""Seal source-scope diagnostics; never promotes a full retrieval checkpoint."""
import json
import shutil
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    output = Path('paper/artifacts/source_scope_stage_seal_20260916.json')
    if output.exists():
        raise FileExistsError(output)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    snapshot = base/'source_scope_stage_snapshot_20260916'
    snapshot.mkdir(exist_ok=False)
    sources = ['radio_gs/training/source_region_hierarchy.py','radio_gs/training/source_region_tracks.py',
               'radio_gs/scripts/compile_three_query_episodes.py','scripts/audits/evaluate_source_scope_fusion.py',
               'scripts/audits/audit_scope_consistent_episodes.py','scripts/audits/seal_source_scope_stage.py',
               'tests/test_source_region_hierarchy.py','tests/test_source_region_tracks.py']
    source_receipts = {}
    for source in sources:
        target = snapshot/source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        digest = sha256_file(source)
        assert digest==sha256_file(target)
        source_receipts[source]=dict(sha256=digest,snapshot=str(target))
    identities = {'figurines':'502469ad7f43febf39022aa2bc8d5c78aed57d0d662a14ee2090b994c86c7561',
                  'ramen':'d86d5d324500c45ec74a7b35b55916be09da01cc5aa0e9bbbe89f684f7fa1856'}
    artifacts = {}
    for scene, identity in identities.items():
        root=base/'preserved_source_inventory_20260916'/scene
        fp=root/'exact_render_query_threshold96/field.pt'
        assert sha256_file(fp)==identity
        field=torch.load(fp,weights_only=False)
        path=Path(f'paper/artifacts/{scene}_source_scope_fusion_20260916.json')
        result=json.loads(path.read_text())
        assert result['implementation_sha256']==source_receipts['scripts/audits/evaluate_source_scope_fusion.py']['sha256']
        controls=Path(result['source_controls_path'])
        assert sha256_file(controls)==result['source_controls_sha256']
        state=torch.load(controls,weights_only=False)
        assert state['geometry_binding']==field['geometry_binding']
        assert torch.equal(state['valid'],field['state_dict']['observed'])
        torch.testing.assert_close(state['latents']['parent_native'][state['valid']],field['state_dict']['latent'][state['valid'],:64],atol=2e-5,rtol=2e-4)
        reference=json.loads(Path(f'paper/artifacts/{scene}_local_geometry_native_rebuild_20260916.json').read_text())
        for key in ['iou','average_precision']:
            assert abs(result['macro']['parent_native'][key]-reference['macro']['parent_native'][key])<1e-5
        episodes=base/'scope_consistent_episodes_20260916'/scene/'episodes.pt'
        for asset in [fp,path,controls,episodes,episodes.with_suffix('.json')]:
            artifacts[str(asset.resolve())]=sha256_file(asset)
    audit=Path('paper/artifacts/source_scope_consistency_audit_20260916.json')
    artifacts[str(audit.resolve())]=sha256_file(audit)
    output.write_text(json.dumps(dict(schema='source_scope_stage_seal.v1',sources=source_receipts,artifacts=artifacts,
          complete_mainline_unchanged=True,native_scope_controls_rejected=True,new_supervision_compiled=True,
          new_supervision_full_training_complete=False,physical_identity_verified=False,
          new_full_method_retrieval_metrics=False,relevant_tests_passed=41),indent=2)+'\n')
    print(output, 'sealed',len(artifacts),'artifacts',len(sources),'sources')


if __name__=='__main__':main()
