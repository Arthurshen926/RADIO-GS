"""Seal matched visibility controls, preserving the retained retrieval model."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    out=Path('paper/artifacts/visibility_extent_seal_20260916.json');assert not out.exists()
    archive=Path('/mnt/pool/sqy/results/RADIO-GS/output/visibility_extent_sources_20260916');archive.mkdir(exist_ok=True)
    sources=['scripts/audits/diagnose_source_mask_mass.py','scripts/audits/diagnose_depth_matched_mask_mass.py','scripts/audits/diagnose_direct_depth_mask_transfer.py','scripts/audits/refit_visibility_weighted_extents.py','scripts/audits/visualize_source_white_napkin.py','scripts/audits/seal_visibility_extent_controls.py','scripts/audits/evaluate_frozen_field_lerf_labels.py','radio_gs/training/source_membership_consensus.py','radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py','tests/test_source_membership_consensus.py']
    snaps=[]
    for source in sources:
        path=Path(source);digest=sha256_file(path);dest=archive/(digest+'.py')
        if not dest.exists():dest.write_bytes(path.read_bytes())
        assert sha256_file(dest)==digest;snaps.append(dict(path=source,sha256=digest,snapshot=str(dest)))
    paths=[];checks={}
    for scene in ['figurines','ramen']:
        root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/scene
        pp=root/'exact_render_query_threshold96/field.pt';parent=torch.load(pp,weights_only=False);assert sha256_file(pp)==json.loads((pp.parent/'report.json').read_text())['field_sha256']
        st=parent['architecture']['grounded_transfer']['identity_selection']['trusted_concept_start']
        newpath=Path(f'paper/artifacts/{scene}_visibility_extent_text_20260916.json');new=json.loads(newpath.read_text())
        # Compare against prior sealed exact-calibration evaluation wherever found.
        originals=[]
        for path in Path('paper/artifacts').glob(scene+'*exact*text*20260916.json'):
            data=json.loads(path.read_text())
            if 'exact_render_query_threshold96' in data.get('runs',{}):originals.append(data['runs']['exact_render_query_threshold96'])
        for prior in originals:assert prior['observations']==new['runs']['exact_render_query_threshold96']['observations']
        for mode in ['equal','mass']:
            path=root/('source_visibility_'+mode+'_extent96')/'field.pt';report=json.loads((path.parent/'report.json').read_text());assert sha256_file(path)==report['field_sha256'];candidate=torch.load(path,weights_only=False)
            assert parent['architecture']==candidate['architecture'] and parent['geometry_binding']==candidate['geometry_binding']
            for key,value in parent['state_dict'].items():
                if key in report['changed_tensor_keys']:assert torch.equal(value[:st],candidate['state_dict'][key][:st])
                else:assert torch.equal(value,candidate['state_dict'][key]),key
            paths.extend([path,path.parent/'report.json'])
        for suffix in ['source_mask_mass','direct_depth_mask_transfer','depth_matched_mask_mass','visibility_extent_text']:
            paths.append(Path(f'paper/artifacts/{scene}_{suffix}_20260916.json'))
        paths.append(newpath.with_suffix('.protocol.json'))
        checks[scene]=dict(retained_field_sha256=sha256_file(pp),native_and_query_and_routing_state_exact=True,independent_heads_exact=True,candidates_rejected=True,prior_eval_files_compared=len(originals))
    paths.extend([Path('paper/artifacts/direct_depth_matched_mask_comparison_20260916.json'),Path('/mnt/pool/sqy/results/RADIO-GS/output/source_white_napkin_atlas_20260916.png')])
    out.write_text(json.dumps(dict(schema='visibility_extent_seal.v1',sources=snaps,artifacts=[dict(path=str(p),sha256=sha256_file(p)) for p in paths],checks=checks),indent=2)+'\n');print('sealed',len(snaps),'sources',len(paths),'artifacts',checks)


if __name__=='__main__':main()
