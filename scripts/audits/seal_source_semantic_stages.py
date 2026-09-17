"""Seal source-stage diagnostics and rejected native render-inversion control."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    destination=Path('paper/artifacts/source_semantic_stages_seal_20260916.json')
    if destination.exists():raise FileExistsError(destination)
    archive=Path('/mnt/pool/sqy/results/RADIO-GS/output/source_semantic_stages_sources_20260916');archive.mkdir(exist_ok=True)
    sources=['scripts/audits/diagnose_source_semantic_stages.py','scripts/audits/diagnose_source_mask_transfer.py',
             'scripts/audits/fit_render_consistent_native_fusion.py','radio_gs/training/render_consistent_fusion.py',
             'tests/test_render_consistent_fusion.py','scripts/audits/evaluate_frozen_field_lerf_labels.py',
             'scripts/audits/seal_source_semantic_stages.py']
    snapshots=[]
    for source in sources:
        path=Path(source);digest=sha256_file(path);target=archive/(digest+'.py')
        if not target.exists():target.write_bytes(path.read_bytes())
        assert sha256_file(target)==digest
        snapshots.append(dict(source=source,sha256=digest,snapshot=str(target)))
    artifacts=[];checks={}
    for scene in ['figurines','ramen']:
        root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/scene
        paths=[Path(f'paper/artifacts/{scene}_{suffix}_20260916.json') for suffix in [
            'source_semantic_stages','source_semantic_stages_common','source_mask_transfer',
            'render_consistent_native_text','render_consistent_native_text_20260916.protocol'.replace('_20260916','')]]
        # Protocol has .protocol before .json, rather than before date.
        paths[-1]=Path(f'paper/artifacts/{scene}_render_consistent_native_text_20260916.protocol.json')
        parent=root/'full_source_query_threshold_control/field.pt';candidate=root/'render_consistent_native_control/field.pt'
        pr=json.loads((parent.parent/'report.json').read_text());cr=json.loads((candidate.parent/'report.json').read_text())
        assert sha256_file(parent)==pr['field_sha256'] and sha256_file(candidate)==cr['field_sha256']
        before=torch.load(parent,weights_only=False);after=torch.load(candidate,weights_only=False)
        assert before['architecture']==after['architecture'] and before['geometry_binding']==after['geometry_binding']
        for key,value in before['state_dict'].items():
            if key!='latent':assert torch.equal(value,after['state_dict'][key]),key
        assert torch.equal(before['state_dict']['latent'][:,64:],after['state_dict']['latent'][:,64:])
        observed=before['state_dict']['observed']
        assert torch.equal(before['state_dict']['latent'][~observed],after['state_dict']['latent'][~observed])
        assert cr['metrics']['source_mse_after']<cr['metrics']['source_mse_before']
        new=json.loads(paths[3].read_text())['runs']['full_source_query_threshold_control']
        old=json.loads(Path(f'paper/artifacts/{scene}_full_source_query_threshold_text_20260916.json').read_text())['runs']['full_source_query_threshold_control']
        assert new['observations']==old['observations']
        retained=root/'exact_render_query_threshold96/field.pt'
        assert sha256_file(retained)==json.loads((retained.parent/'report.json').read_text())['field_sha256']
        for path in paths+[candidate,candidate.parent/'report.json']:
            artifacts.append(dict(path=str(path),sha256=sha256_file(path)))
        checks[scene]=dict(parent_observations_exact=True,all_nonprefix_state_exact=True,unobserved_state_exact=True,
            retained_mainline_sha256=sha256_file(retained),candidate_rejected=True)
    destination.write_text(json.dumps(dict(schema='source_semantic_stages_seal.v1',sources=snapshots,artifacts=artifacts,checks=checks,
        conclusion='Source mask cross-view transfer is weak even without semantic features. Source feature reconstruction MSE reduction failed real native retrieval; no mainline promotion.'),indent=2)+'\n')
    print('sealed',len(snapshots),'sources',len(artifacts),'artifacts',flush=True)


if __name__=='__main__':main()
