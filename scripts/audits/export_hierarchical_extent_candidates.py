"""Keep verified concept extents alongside fine source-region alternatives.

Two global residual heads are represented exactly by a wider block-diagonal
head over the same frozen element vector, not by a second element map.
"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--root',type=Path);a=p.parse_args();torch.set_num_threads(4)
    root=a.root if a.root is not None else Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/a.scene
    parents=[];reports=[]
    for name in ['source_region_extent96','source_extent_rendered_residual96']:
        r=json.loads((root/name/'report.json').read_text());assert sha256_file(root/name/'field.pt')==r['field_sha256']
        parents.append(torch.load(root/name/'field.pt',weights_only=False));reports.append(r)
    region,concept=parents;assert region['geometry_binding']==concept['geometry_binding']
    for k,v in region['state_dict'].items():
        if not k.startswith('grounded_transfer.'):assert torch.equal(v,concept['state_dict'][k]),k
    R=reports[0]['proposal_count'];G=len(reports[0]['groups'])
    architecture=dict(region['architecture'],grounded_transfer=dict(region['architecture']['grounded_transfer'],num_proposals=R+G,residual_dim=256))
    field=ElementRetrievalField(**architecture)
    keep={k:v for k,v in region['state_dict'].items() if not k.startswith(('grounded_transfer.decoder.','grounded_transfer.residual.','grounded_transfer.proposal_'))}
    missing=field.load_state_dict(keep,strict=False);assert not missing.unexpected_keys
    h=field.grounded_transfer;rs=region['state_dict'];cs=concept['state_dict'];prefix='grounded_transfer.'
    with torch.no_grad():
        for name in ['weight','bias']:
            getattr(h.decoder,name).copy_(torch.cat([rs[prefix+'decoder.'+name],cs[prefix+'decoder.'+name]],0))
            getattr(h.residual[0],name).copy_(torch.cat([rs[prefix+'residual.0.'+name],cs[prefix+'residual.0.'+name]],0))
        h.residual[2].weight.zero_()
        h.residual[2].weight[:R,:128].copy_(rs[prefix+'residual.2.weight'])
        h.residual[2].weight[R:,128:].copy_(cs[prefix+'residual.2.weight'])
        h.residual[2].bias.copy_(torch.cat([rs[prefix+'residual.2.bias'],cs[prefix+'residual.2.bias']]))
        h.proposal_groups[:R].copy_(rs[prefix+'proposal_groups']);h.proposal_groups[R:].copy_(torch.eye(G,dtype=torch.bool))
        h.proposal_embeddings[:R].copy_(rs[prefix+'proposal_embeddings'])
        for j in range(G):h.proposal_embeddings[R+j].copy_(F.normalize(rs[prefix+'proposal_embeddings'][rs[prefix+'proposal_groups'][:,j]].float().mean(0),dim=0))
        rows=torch.arange(0,len(field.latent),137);z=h.features(field.latent[rows]);actual=h.decoder(z)+h.residual(z)
        for offset,payload in [(0,region),(R,concept)]:
            old=ElementRetrievalField(**payload['architecture']);old.load_state_dict(payload['state_dict']);oh=old.grounded_transfer
            expected=oh.decoder(z)+oh.residual(z)
            torch.testing.assert_close(actual[:,offset:offset+expected.shape[1]],expected,rtol=1e-5,atol=1e-6)
    path=root/'caption_region_evidence.pt';assert sha256_file(path)==reports[0]['evidence_sha256'];e=torch.load(path,weights_only=False)
    origins=e['proposal_views'][reports[0]['proposal_indices']].tolist()+[-1]*G
    out=root/'source_hierarchical_candidates96';out.mkdir(exist_ok=False)
    torch.save(dict(region,architecture=architecture,state_dict=field.state_dict()),out/'field.pt')
    r=dict(reports[0],schema='hierarchical_extent_candidates.v1',field_sha256=sha256_file(out/'field.pt'),
           parent_field_sha256=reports[0]['field_sha256'],concept_parent_sha256=reports[1]['field_sha256'],
           proposal_count=R+G,proposal_indices=reports[0]['proposal_indices']+[-1]*G,proposal_origins=origins,
           independent_region_count=R,concept_candidate_count=G,scope='untrained selector input; not a promoted retrieval result',
           construction='exact block-diagonal residual head; mean native region descriptor for each concept candidate',
           implementation_sha256=sha256_file(__file__),readout_implementation_sha256=sha256_file('radio_gs/models/grounded_semantic_transfer.py'))
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(a.scene,R,G)


if __name__=='__main__':main()
