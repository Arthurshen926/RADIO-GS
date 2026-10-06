"""Canonical source concept encoding, using source SAM phrases, not target queries."""
import json
from pathlib import Path
import torch
from radio_gs.models.functional_query_encoder import FunctionalThreeQueryEncoder
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
    torch.set_num_threads(4);encoder=None;last_identity=None
    for scene in ['figurines','ramen','teatime','waldo_kitchen']:
        root=Path('output/full_field_rebuild_next19')/scene/'reconstruction/field';out=Path('output/source_uplift_scope_next20')/scene/'plain_queries.pt';assert not out.exists()
        identity=json.loads((root/'native_query_encoder.json').read_text())
        if encoder is None or identity!=last_identity:
            encoder=FunctionalThreeQueryEncoder(identity,device='cuda:0');last_identity=identity
        raw=torch.load(root/'source_grounded_text.pt',map_location='cpu',weights_only=False);ids=json.loads((root/'exact_render_query_threshold96/report.json').read_text())['query_indices'];assert not raw['benchmark_labels_used']
        phrases=[raw['grounding_phrases'][i] for i in ids]
        assert phrases==[raw['queries'][i].removeprefix('a photo of a ') for i in ids]
        queries=encoder.encode_text(phrases).half().float().cpu()
        torch.save(dict(queries=queries,phrases=phrases,query_ids=ids,source_only=True,encoder_identity=identity,raw_source_sha256=sha256_file(root/'source_grounded_text.pt'),implementation_sha256=sha256_file(__file__)),out)
        print(scene,phrases,flush=True)
if __name__=='__main__':main()
