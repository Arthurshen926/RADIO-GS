"""Seal source-evidence controls and query-boundary development results."""
import json
from pathlib import Path
import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    base=Path('/mnt/pool/sqy/results/RADIO-GS/output')
    archive=base/'source_evidence_localization_sources_20260916';archive.mkdir(exist_ok=True)
    sources={}
    for name in [__file__,'scripts/audits/export_source_fusion_controls.py',
                 'scripts/audits/diagnose_region_pooling.py','scripts/audits/export_dense_native_anchor.py',
                 'scripts/audits/fit_native_query_threshold.py','scripts/audits/fit_full_source_query_threshold.py',
                 'radio_gs/models/native_query_threshold.py','radio_gs/models/element_retrieval_field.py',
                 'scripts/audits/evaluate_frozen_field_lerf_labels.py','scripts/audits/evaluate_real_reference_queries.py']:
        p=Path(name);digest=sha256_file(p);snapshot=archive/(digest+'.py')
        if not snapshot.exists():snapshot.write_bytes(p.read_bytes())
        assert sha256_file(snapshot)==digest;sources[str(p)]=digest
    result=dict(schema='source_evidence_localization.v1',source_files=sources,scenes={},artifacts={},
                status='full source query threshold is a scoped text development candidate; no joint benchmark promotion',
                no_new_semantic_ranking_breakthrough=True)
    parent_name='scale_affinity_v1_direct_control'
    modes=['source_fusion_queryfree_control','source_fusion_view_balanced_control','dense_local_native_control',
           'native_query_threshold_control','full_source_query_threshold_control']
    for scene in ['figurines','ramen']:
        root=base/'preserved_source_inventory_20260916'/scene
        parent=torch.load(root/parent_name/'field.pt',weights_only=False)
        fields={parent_name:sha256_file(root/parent_name/'field.pt')}
        for name in modes:
            p=root/name/'field.pt';digest=sha256_file(p);report=json.loads((p.parent/'report.json').read_text())
            assert digest==report['field_sha256'] and report['parent_field_sha256']==fields[parent_name]
            assert report['benchmark_data_used'] is False
            fields[name]=digest
            assert (archive/(report['implementation_sha256']+'.py')).exists()
            payload=torch.load(p,weights_only=False)
            for key,value in parent['state_dict'].items():
                if 'query_threshold' in name or key!='latent':assert torch.equal(value,payload['state_dict'][key]),(name,key)
            assert torch.equal(parent['state_dict']['latent'][:,64:],payload['state_dict']['latent'][:,64:])
            result['artifacts'][str(p.parent/'report.json')]=sha256_file(p.parent/'report.json')
        artifacts={}
        for suffix in ['source_fusion_bias_text','source_fusion_bias_reference','region_pooling_diagnostic',
                       'dense_local_native_text','dense_local_native_reference','native_query_threshold_text',
                       'full_source_query_threshold_text','native_query_threshold_reference']:
            p=Path(f'paper/artifacts/{scene}_{suffix}_20260916.json')
            result['artifacts'][str(p)]=sha256_file(p);artifacts[suffix]=json.loads(p.read_text())
        baseline=artifacts['source_fusion_bias_text']['runs'][parent_name]
        summary={'parent':{k:baseline[k] for k in ['observation_miou','average_precision_macro']}}
        for suffix in ['source_fusion_bias_text','dense_local_native_text','native_query_threshold_text','full_source_query_threshold_text']:
            for name,run in artifacts[suffix]['runs'].items():
                assert run['field_sha256']==fields[name]
                summary[name]={k:run[k] for k in ['observation_miou','average_precision_macro']}
        original=artifacts['source_fusion_bias_reference']['runs'][parent_name]
        for name,run in artifacts['native_query_threshold_reference']['runs'].items():
            assert run['field_sha256']==fields[name]
            for old,new in zip(original['observations'],run['observations']):
                for key in ['reference','target_frame','category','image','prompt']:assert old[key]==new[key],(scene,name,key)
            assert len(original['observations'])==len(run['observations'])
        # Independently cold-load deployed fields and exercise native/image/prompt.
        gr=json.loads((base/'geometry_only_repair_final_20260914'/scene/'report.json').read_text())['results']['independent_scale']
        geo=FrozenGaussianGeometry.load(gr['geometry'],expected_sha256=gr['geometry_sha256'],device='cpu')
        identity=json.loads((root/'native_query_encoder.json').read_text())
        old,_=load_element_retrieval_field(root/parent_name/'field.pt',expected_sha256=fields[parent_name],geometry=geo,expected_query_encoder=identity,device='cpu')
        data=torch.load(root/'episodes.pt',weights_only=False)
        with torch.no_grad():
            for name in ['native_query_threshold_control','full_source_query_threshold_control']:
                new,_=load_element_retrieval_field(root/name/'field.pt',expected_sha256=fields[name],geometry=geo,expected_query_encoder=identity,device='cpu')
                for modality in ['image','prompt']:
                    q=data['queries'][modality][:2].float()
                    assert torch.equal(old.retrieve(q,modality)['scores'],new.retrieve(q,modality)['scores'])
                q=data['queries']['text'][:3].float()
                assert torch.equal(old.native_cosine(q),new.native_cosine(q))
        result['scenes'][scene]=dict(fields=fields,text=summary,
            diagnostic=artifacts['region_pooling_diagnostic']['macro'],
            cold_native_cosine_image_prompt_exact=True,reference_image_prompt_observations_exact=True)
    path=Path('paper/artifacts/source_evidence_localization_20260916.json')
    path.write_text(json.dumps(result,indent=2)+'\n');print(path)


if __name__=='__main__':main()
