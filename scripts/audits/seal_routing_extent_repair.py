"""Seal paired routing, supervision and resolution controls without promotion."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
    archive=base/'routing_extent_sources_20260916';archive.mkdir(exist_ok=True)
    files=['radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py',
           'tests/test_grounded_semantic_transfer.py','tests/test_rendered_extent_objective.py']
    files += ['scripts/audits/'+s+'.py' for s in ['export_source_routing_controls','audit_source_routing',
        'train_rendered_extent','train_native_mask_extent','train_source_region_extents','diagnose_extent_routing_ceiling',
        'evaluate_frozen_field_lerf_labels','evaluate_real_reference_queries','seal_routing_extent_repair']]
    sources={}
    for name in files:
        path=Path(name);digest=sha256_file(path);snapshot=archive/(digest+'.py')
        if not snapshot.exists():snapshot.write_bytes(path.read_bytes())
        assert sha256_file(snapshot)==digest;sources[name]=digest
    def implementation(digest):
        path=archive/(digest+'.py')
        assert path.exists() and sha256_file(path)==digest,digest
    result=dict(schema='routing_extent_repair.v1',source_files=sources,artifacts={},scenes={},
        status='paired development controls; no independent benchmark promotion')
    names=['source_route_centered96','source_route_competition96','source_route_centered_competition96',
           'source_extent_lifted96','source_extent_rendered96','source_extent_lifted_residual96','source_extent_rendered_residual96',
           'source_native_mask_w128_96','source_native_mask_w480_96','source_region_extent96']
    for scene in ['figurines','ramen']:
        root=base/'preserved_source_inventory_20260916'/scene
        parent=root/'grounded_semantic_transfer96/field.pt';old=torch.load(parent,weights_only=False)
        fields={'grounded_semantic_transfer96':sha256_file(parent)}
        for name in names:
            path=root/name/'field.pt';report=json.loads((path.parent/'report.json').read_text())
            assert sha256_file(path)==report['field_sha256'] and report['parent_field_sha256']==fields['grounded_semantic_transfer96']
            assert report['source_only'] and not report['benchmark_data_used']
            implementation(report['implementation_sha256']);implementation(report['readout_implementation_sha256'])
            new=torch.load(path,weights_only=False)
            assert new['geometry_binding']==old['geometry_binding'] and new['query_encoder']==old['query_encoder']
            for k,v in old['state_dict'].items():
                if name.startswith('source_route_'):
                    if k=='grounded_transfer.acceptance':continue
                elif k.startswith('grounded_transfer.decoder.'):continue
                assert torch.equal(v,new['state_dict'][k]),(name,k)
            fields[name]=report['field_sha256'];del new
            result['artifacts'][str(path.parent/'report.json')]=sha256_file(path.parent/'report.json')
        summary={}
        for suffix in ['source_routing_text','source_extent_objective_text','source_extent_residual_text','native_mask_resolution_text','source_region_extent_text','extent_repair_reference','source_region_extent_reference']:
            path=Path(f'paper/artifacts/{scene}_{suffix}_20260916.json');data=json.loads(path.read_text())
            assert sha256_file(path.with_suffix('.protocol.json'))==data['protocol_sha256']
            implementation(data['protocol']['implementation_sha256'])
            for name,run in data['runs'].items():
                assert run['field_sha256']==fields[name]
                if 'observation_miou' in run:summary[name]={k:run[k] for k in ['observation_miou','average_precision_macro']}
            result['artifacts'][str(path)]=sha256_file(path)
            result['artifacts'][str(path.with_suffix('.protocol.json'))]=sha256_file(path.with_suffix('.protocol.json'))
            if suffix.endswith('_reference'):
                baseline=json.loads(Path(f'paper/artifacts/{scene}_grounded_transfer_reference_20260916.json').read_text())['runs']['grounded_semantic_transfer96']
                for run in data['runs'].values():
                    assert len(baseline['observations'])==len(run['observations'])
                    for before,after in zip(baseline['observations'],run['observations']):
                        for k in ['reference','target_frame','category','image','prompt']:assert before[k]==after[k]
        for suffix in ['source_routing_audit','extent_capacity_oracle','extent_region_oracle','learned_region_oracle','region_selection_oracle']:
            path=Path(f'paper/artifacts/{scene}_{suffix}_20260916.json');data=json.loads(path.read_text());implementation(data['implementation_sha256'])
            result['artifacts'][str(path)]=sha256_file(path)
        oracle=json.loads(Path(f'paper/artifacts/{scene}_extent_capacity_oracle_20260916.json').read_text())
        native=json.loads(Path(f'paper/artifacts/{scene}_caption_transfer_text_20260916.json').read_text())['runs']['full_source_query_threshold_control']
        lookup={(r['frame'],r['category']):r['iou'] for r in native['observations']}
        ceiling={k:sum(max(r[k]['iou'],lookup[r['frame'],r['category']]) for r in oracle['observations'])/len(oracle['observations']) for k in oracle['macro']}
        result['scenes'][scene]=dict(fields=fields,text=summary,all_parent_field_tensors_exact=True,reference_image_prompt_exact=True,
            oracle_including_native_fallback=ceiling,oracle_is_not_a_deployable_score=True)
    path=Path('paper/artifacts/routing_extent_repair_seal_20260916.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n');print(path)


if __name__=='__main__':main()
