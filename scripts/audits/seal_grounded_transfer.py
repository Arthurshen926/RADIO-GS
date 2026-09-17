"""Bind source-discovery controls, immutable implementations and dev evaluation."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    archive = base/'caption_transfer_sources_20260916'
    archive.mkdir(exist_ok=True)
    names = ['discover_source_caption_phrases', 'ground_source_caption_phrases',
             'encode_caption_region_evidence', 'export_caption_fusion_control',
             'export_source_membership_code', 'fit_grounded_semantic_transfer',
             'train_grounded_semantic_transfer', 'build_fixed_vocab_transfer_control',
             'evaluate_frozen_field_lerf_labels', 'evaluate_real_reference_queries',
             'audit_grounded_transfer', 'seal_grounded_transfer']
    sources = [Path('scripts/audits')/(n+'.py') for n in names]
    sources += [Path('radio_gs/models')/(n+'.py') for n in
                ['element_retrieval_field', 'grounded_semantic_transfer', 'source_membership_code', 'native_query_threshold']]
    snapshots = {}
    for source in sources:
        digest = sha256_file(source)
        target = archive/(digest+'.py')
        if not target.exists():
            target.write_bytes(source.read_bytes())
        assert sha256_file(target) == digest
        snapshots[str(source)] = digest
    def implementation(digest):
        target = archive/(digest+'.py')
        assert target.exists() and sha256_file(target) == digest, digest
    result = dict(schema='source_grounded_transfer_seal.v1', source_files=snapshots, scenes={}, artifacts={},
                  status='linear gated source-discovered extent is a LERF development candidate; no joint benchmark promotion')
    variants = ['caption_native_fusion_control', 'caption_coarse_fusion_control',
                'source_membership_code32', 'source_membership_balanced32',
                'grounded_semantic_transfer96', 'grounded_semantic_transfer_mlp96', 'fixed_vocab_grounded_transfer96']
    for scene in ['figurines', 'ramen']:
        root = base/'preserved_source_inventory_20260916'/scene
        info = json.loads((root/'teachers.json').read_text())
        source = json.loads((root/'source_frames.json').read_text())
        captions = json.loads((root/'caption_source_phrases.json').read_text())
        assert captions['source_only'] and not captions['benchmark_vocabulary_used']
        assert captions['teacher_manifest_sha256'] == sha256_file(root/'teachers.json')
        assert captions['train_frames'] == info['train_frames']
        sources_by_frame = {s['frame_idx']: s for s in source['frames']}
        for record in captions['records']:
            assert record['frame'] in info['train_frames']
            src = sources_by_frame[record['frame']]
            assert record['rgb_sha256'] == src['source_sha256']
        for frame in info['train_frames']:
            src = sources_by_frame[frame]
            assert sha256_file(Path(source['image_dir'])/src['source_file']) == src['source_sha256']
        for model, digest in captions['model_files'].items():
            assert sha256_file(model) == digest
        implementation(captions['implementation_sha256'])
        for asset in ['caption_grounded_regions', 'caption_region_evidence', 'fixed_vocab_region_evidence']:
            receipt = json.loads((root/(asset+'.json')).read_text())
            assert sha256_file(root/(asset+'.pt')) == receipt['sha256']
            implementation(receipt['implementation_sha256'])
            result['artifacts'][str(root/(asset+'.json'))] = sha256_file(root/(asset+'.json'))
        parent = root/'full_source_query_threshold_control'/'field.pt'
        parent_state = torch.load(parent, weights_only=False)
        fields = {'full_source_query_threshold_control': sha256_file(parent)}
        for name in variants:
            path = root/name/'field.pt'
            report = json.loads((path.parent/'report.json').read_text())
            assert sha256_file(path) == report['field_sha256']
            assert report['source_only'] and report['benchmark_data_used'] is False
            implementation(report['implementation_sha256'])
            if 'readout_implementation_sha256' in report:
                implementation(report['readout_implementation_sha256'])
            fields[name] = report['field_sha256']
            if name in ['grounded_semantic_transfer96', 'grounded_semantic_transfer_mlp96', 'fixed_vocab_grounded_transfer96']:
                payload = torch.load(path, weights_only=False)
                assert payload['geometry_binding'] == parent_state['geometry_binding']
                assert report['parent_field_sha256'] == fields['full_source_query_threshold_control']
                for key, value in parent_state['state_dict'].items():
                    assert torch.equal(value, payload['state_dict'][key]), (name, key)
                del payload
            result['artifacts'][str(path.parent/'report.json')] = sha256_file(path.parent/'report.json')
        summary = {}
        for suffix in ['caption_transfer_text', 'caption_transfer_reference', 'grounded_transfer_text',
                       'grounded_transfer_reference', 'fixed_vocab_transfer_text']:
            path = Path(f'paper/artifacts/{scene}_{suffix}_20260916.json')
            evaluation = json.loads(path.read_text())
            protocol_path = path.with_suffix('.protocol.json')
            assert sha256_file(protocol_path) == evaluation['protocol_sha256']
            implementation(evaluation['protocol']['implementation_sha256'])
            for name, run in evaluation['runs'].items():
                assert run['field_sha256'] == fields[name]
                if 'observation_miou' in run:
                    summary[name] = {k: run[k] for k in ['observation_miou', 'average_precision_macro']}
            if 'annotation_frames' in evaluation:
                assert not set(evaluation['annotation_frames']) & set(info['train_frames'])
            result['artifacts'][str(path)] = sha256_file(path)
            result['artifacts'][str(protocol_path)] = sha256_file(protocol_path)
        path = Path(f'paper/artifacts/{scene}_grounded_transfer_audit_20260916.json')
        audit = json.loads(path.read_text())
        for key in ['all_parent_tensors_exact', 'cold_native_cosine_exact', 'cold_image_response_exact',
                    'reference_image_prompt_observations_exact', 'fallback_query_response_exact', 'unsupported_scores_zero']:
            assert audit[key]
        assert sha256_file(audit['plot']) == audit['plot_sha256']
        implementation(audit['implementation_sha256'])
        result['artifacts'][str(path)] = sha256_file(path)
        result['artifacts'][audit['plot']] = audit['plot_sha256']
        result['artifacts'][str(root/'caption_source_phrases.json')] = sha256_file(root/'caption_source_phrases.json')
        result['scenes'][scene] = dict(fields=fields, text=summary,
            accepted_categories=sum(g['accepted'] for g in audit['gates']), categories=len(audit['gates']),
            all_parent_tensors_exact=True, cold_and_reference_invariants_passed=True)
    path = Path('paper/artifacts/source_grounded_transfer_seal_20260916.json')
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
