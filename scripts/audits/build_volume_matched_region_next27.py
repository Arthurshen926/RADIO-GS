"""Frozen-parent element-count control for query-free region ranking."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.querying.volume_matched_region_readout import match_element_count
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();torch.set_num_threads(4)
 src=Path('output/query_free_region_readout_next26')/a.scene
 pre=json.loads((src/'preparation.json').read_text());assert pre['complete'] and sha256_file(src/'scores.pt')==pre['scores_sha256']
 data=torch.load(src/'scores.pt',map_location='cpu',weights_only=False)
 base=data['scores']['final'];proposed=data['scores']['region_mean'];observed=data['observed']
 matched,records=match_element_count(proposed,base,observed)
 fieldpath=Path('output/full_field_rebuild_next19')/a.scene/'reconstruction/field/bounded_fallback_admission/field.pt'
 assert sha256_file(fieldpath)==pre['field_sha256']
 payload=torch.load(fieldpath,map_location='cpu',weights_only=False)
 model=ElementRetrievalField(**payload['architecture']);model.load_state_dict(payload['state_dict']);model.requires_grad_(False)
 bankpath=Path('output/full_field_rebuild_next19')/a.scene/'evaluation/text_online.encodings.pt'
 bank=torch.load(bankpath,map_location='cpu',weights_only=False)[0]
 q=F.normalize(bank['encoding'].half().float(),dim=-1)
 _,accepted,_=model.grounded_transfer.route(q)
 fullscorepath=Path('output/full_field_rebuild_next19')/a.scene/'evaluation/scores.pt'
 fullscore=torch.load(fullscorepath,map_location='cpu',weights_only=False)
 assert torch.equal(fullscore['scores']['final'],base)
 used=(base-fullscore['scores']['native_compact_calibrated']).abs().amax(0)>2e-5
 admitted=accepted&used
 mixed=torch.where(admitted[None],base,matched)
 scores=dict(final=base,area_matched=matched,area_matched_unrouted=mixed)
 out=Path('output/volume_matched_region_next27')/a.scene;out.mkdir(parents=True,exist_ok=False)
 torch.save(dict(data,scores=scores),out/'scores.pt')
 receipt=dict(complete=True,source_region_preparation_sha256=sha256_file(src/'preparation.json'),source_scores_sha256=pre['scores_sha256'],scores_sha256=sha256_file(out/'scores.pt'),field_sha256=pre['field_sha256'],source_views=pre['source_views'],geometry_binding=data['geometry_binding'],arms=list(scores),parent_count_prior_only=True,benchmark_query_text_used=True,benchmark_gt_used=False,target_rgb_used=False,threshold_fit_on_gt=False,element_ranking_unchanged_per_query=True,pixel_ranking_may_change_after_rendering=True,query_count=len(records),exact_count_queries=sum(x['new_count']==x['parent_count'] for x in records),ties=sum(x['tie'] for x in records),abstentions=sum(x['abstained'] for x in records),routed_queries=int(admitted.sum()),unrouted_queries=int((~admitted).sum()),records=records,implementation_sha256=sha256_file(__file__),default_promoted=False)
 (out/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(a.scene,'exact count',receipt['exact_count_queries'],'/',len(records),'ties',receipt['ties'],flush=True)

if __name__=='__main__':main()
