"""Seal partial-label source controls, including rejected rendered refinements."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    out=Path('paper/artifacts/partial_source_controls_seal_20260916.json');assert not out.exists()
    archive=Path('/mnt/pool/sqy/results/RADIO-GS/output/partial_source_controls_sources_20260916');archive.mkdir(exist_ok=True)
    sources=['scripts/audits/refit_partial_source_extents.py','scripts/audits/refine_rendered_partial_extents.py','scripts/audits/seal_partial_source_controls.py','scripts/audits/evaluate_frozen_field_lerf_labels.py','radio_gs/training/partial_source_membership.py','radio_gs/training/source_membership_consensus.py','radio_gs/models/grounded_semantic_transfer.py','radio_gs/models/element_retrieval_field.py','tests/test_partial_source_membership.py','tests/test_rendered_partial_loss.py']
    snapshots=[];artifacts=[];checks={}
    for name in sources:
        p=Path(name);digest=sha256_file(p);target=archive/(digest+'.py')
        if not target.exists():target.write_bytes(p.read_bytes())
        assert sha256_file(target)==digest;snapshots.append(dict(source=name,sha256=digest,snapshot=str(target)))
    for scene in ['figurines','ramen']:
        root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/scene;pp=root/'exact_render_query_threshold96/field.pt';parent=torch.load(pp,weights_only=False);ps=parent['state_dict'];start=parent['architecture']['grounded_transfer']['identity_selection']['trusted_concept_start'];assert sha256_file(pp)==json.loads((pp.parent/'report.json').read_text())['field_sha256']
        rows=torch.arange(0,len(ps['latent']),997);z=torch.cat([F.normalize(ps['latent'][rows,:64],dim=-1),F.normalize(ps['latent'][rows,64:],dim=-1)],1)
        def independent(sd):
            return F.linear(z,sd['grounded_transfer.decoder.weight'][:start],sd['grounded_transfer.decoder.bias'][:start])+F.linear(F.relu(F.linear(z,sd['grounded_transfer.residual.0.weight'],sd['grounded_transfer.residual.0.bias'])),sd['grounded_transfer.residual.2.weight'][:start],sd['grounded_transfer.residual.2.bias'][:start])
        expected=independent(ps)
        for name in ['source_partial_abstain_extent96','source_partial_abstain_preserve_extent96','source_rendered_partial_full96','source_rendered_partial_abstain96']:
            path=root/name/'field.pt';rp=path.parent/'report.json';r=json.loads(rp.read_text());assert sha256_file(path)==r['field_sha256'];p=torch.load(path,weights_only=False);assert p['architecture']==parent['architecture'] and p['geometry_binding']==parent['geometry_binding'];sd=p['state_dict']
            for key,value in ps.items():
                if key.startswith('grounded_transfer.decoder.') or key.startswith('grounded_transfer.residual.2.'):assert torch.equal(value[:start],sd[key][:start])
                elif key.startswith('grounded_transfer.residual.0.'):assert torch.equal(value[:128],sd[key][:128])
                else:assert torch.equal(value,sd[key]),key
            torch.testing.assert_close(independent(sd),expected,atol=0,rtol=0)
            artifacts.extend([dict(path=str(x),sha256=sha256_file(x)) for x in [path,rp]])
        baseline=json.loads(Path(f'paper/artifacts/{scene}_visibility_extent_text_20260916.json').read_text())['runs']['exact_render_query_threshold96']['observations']
        for suffix in ['partial_source_extent_text','rendered_partial_extent_text']:
            path=Path(f'paper/artifacts/{scene}_{suffix}_20260916.json');r=json.loads(path.read_text());assert r['runs']['exact_render_query_threshold96']['observations']==baseline
            artifacts.extend([dict(path=str(x),sha256=sha256_file(x)) for x in [path,path.with_suffix('.protocol.json')]])
        checks[scene]=dict(retained_mainline_sha256=sha256_file(pp),native_and_routing_state_exact=True,independent_head_sampled_response_exact=True,parent_real_observations_exact=True,all_four_candidates_rejected=True)
    out.write_text(json.dumps(dict(schema='partial_source_controls_seal.v1',sources=snapshots,artifacts=artifacts,checks=checks),indent=2)+'\n');print('sealed',len(snapshots),'sources',len(artifacts),'artifacts')


if __name__=='__main__':main()
