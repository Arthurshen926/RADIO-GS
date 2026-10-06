"""Score explicit selected-set output using the unchanged shared LERF scorer."""
import argparse
import json
from pathlib import Path

import torch
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.evaluation.lerf_readouts import evaluate_lerf_readouts


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--readout', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--direct-map', action='store_true', help='Score native response and its explicit >0.5 subset baseline; no scope inference')
    a = p.parse_args()
    torch.set_num_threads(4)
    scene = json.loads(Path(a.scene).read_text())
    data = torch.load(a.readout, map_location='cpu', weights_only=False)
    if a.direct_map:
        data = dict(texts=data['texts'], scores=data['semantic_probability'],
                    selection_scores=data['semantic_probability'],
                    semantic_geometry=data['geometry'], selection_geometry=data['geometry'],
                    semantic_observed=data['observed'], selection_observed=data['observed'])
    binding = data['semantic_geometry']
    geometry = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device='cuda')
    selected_binding = data['selection_geometry']
    selected_geometry = FrozenGaussianGeometry.load(selected_binding['path'], expected_sha256=selected_binding['sha256'], device='cuda')
    observed = data['semantic_observed'].cuda()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_lerf_readouts(
        scene=scene['scene'].removeprefix('lerf_'), out=out, texts=data['texts'],
        scores={'hierarchical_pipeline': data['scores'].cuda()},
        selection_scores={'hierarchical_pipeline': data['selection_scores'].cuda()},
        geo=geometry, default_observed=observed, semantic_geo=geometry, semantic_observed=observed,
        selection_geo=selected_geometry, selection_observed=data['selection_observed'].cuda(),
        camera_report=scene['camera_report'], camera_report_sha256=scene['camera_report_sha256'])
    (out/'evaluation.json').write_text(json.dumps(dict(key=scene['scene'], metrics=metrics,
        development=True, readout=a.readout, direct_map_threshold_baseline=a.direct_map,
        protocol='unchanged LERF semantic and selected-projection scorers'), indent=2)+'\n')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
