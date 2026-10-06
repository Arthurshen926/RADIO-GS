"""Run explicit text queries against a declared native map, without old answers.

This is the map semantic baseline, not the legacy calibrated/scope result.
Uses the existing cybersim_agent native encoder environment.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.models.native_query_encoders import load_siglip_text
from radio_gs.querying.native_map_response import native_map_response


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic()
    scene = json.loads(Path(args.scene).read_text())
    queries = json.loads(Path(args.queries).read_text())['queries']
    texts = [q['text'] for q in queries]
    model, tokenizer = load_siglip_text(scene['text_encoder'], args.device)
    tokens = tokenizer(texts, padding='max_length', max_length=64,
                       truncation=True, return_tensors='pt').to(args.device)
    encodings = F.normalize(model(**tokens).pooler_output.float(), dim=-1)
    field = torch.load(scene['native_field'], map_location='cpu', weights_only=False)
    cosine, probability = native_map_response(field, encodings, args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(queries=queries, texts=texts, encodings=encodings.cpu(),
                    cosine=cosine, semantic_probability=probability,
                    geometry=field['geometry'], observed=field['supported'].reshape(-1),
                    output_contract='semantic_response_only_no_selected_set'), output/'response.pt')
    report = dict(scene=scene['scene'], query_count=len(queries), elements=len(probability),
                  feature_dimension=field['features'].shape[1], mode='map_plus_declared_query',
                  native_field=scene['native_field'], text_encoder=scene['text_encoder'],
                  calibration=scene['calibration'], seconds=time.monotonic()-start,
                  historical_predictions_read=False, source_rgb_read=False,
                  target_rgb_read=False, ground_truth_read=False,
                  subset_inference=False, benchmark_metrics=None,
                  queries=[dict(q, maximum_response=float(probability[:, j].max()),
                                positive_elements=int((probability[:, j] > .5).sum()))
                           for j, q in enumerate(queries)])
    (output/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['scene','query_count','elements','seconds']}), flush=True)


if __name__ == '__main__':
    main()
