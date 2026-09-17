"""Use the same evaluation contract with a declared semantic capability provider."""
import argparse
import importlib
import json
import sys
from pathlib import Path
import torch

from radio_gs.models.functional_query_encoder import FunctionalThreeQueryEncoder
from radio_gs.querying.registered_coordinate_query import encode_registered_coordinates, coordinate_only_prompt_architecture
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', required=True, choices=['text', 'reference'])
    parser.add_argument('--coordinate-only-prompt', action='store_true')
    args, remaining = parser.parse_known_args()
    metadata_parser = argparse.ArgumentParser(add_help=False)
    metadata_parser.add_argument('--root', type=Path, required=True)
    metadata_parser.add_argument('--runs', nargs='+', required=True)
    metadata_parser.add_argument('--output', type=Path, required=True)
    metadata, _ = metadata_parser.parse_known_args(remaining)
    dimensions = set()
    if args.coordinate_only_prompt:
        if args.task != 'reference' or '--refine-prompts' in remaining:
            raise ValueError('coordinate-only encoding requires unrefined registered prompt evaluation')
        for run in metadata.runs:
            folder = metadata.root / run
            receipt = json.loads((folder / 'report.json').read_text())
            assert sha256_file(folder / 'field.pt') == receipt['field_sha256']
            payload = torch.load(folder / 'field.pt', weights_only=False, map_location='cpu')
            if not coordinate_only_prompt_architecture(payload['architecture']):
                raise ValueError('a requested model consumes prompt appearance; cannot omit it')
            dimensions.add(payload['architecture']['query_dims']['prompt'])
        if len(dimensions) != 1:
            raise ValueError('inconsistent registered prompt dimensions')

    class EvaluationEncoder(FunctionalThreeQueryEncoder):
        def encode_prompt(self, image_path, seed_pixels, world_xyz, minimum, extent, height, width):
            if not args.coordinate_only_prompt:
                return super().encode_prompt(image_path, seed_pixels, world_xyz, minimum, extent, height, width)
            if len(seed_pixels) != 2 or seed_pixels[0] == seed_pixels[1] or any(x < 0 or x >= height*width for x in seed_pixels):
                raise ValueError('invalid signed registered pixels')
            return encode_registered_coordinates(world_xyz.to(self.device), minimum, extent, next(iter(dimensions)))
    name = ('evaluate_frozen_field_lerf_labels' if args.task == 'text'
            else 'evaluate_real_reference_queries')
    evaluator = importlib.import_module('scripts.audits.' + name)
    evaluator.NativeThreeQueryEncoder = EvaluationEncoder
    sys.argv = [sys.argv[0], *remaining]
    evaluator.main()
    # The underlying evaluator's identity alone would omit this dispatch path.
    path = metadata.output.with_suffix('.dispatch.json')
    path.write_text(json.dumps(dict(
        schema='functional_query_evaluation_dispatch.v1', coordinate_only_prompt=args.coordinate_only_prompt,
        implementation_sources={str(p): sha256_file(p) for p in [Path(__file__), Path('radio_gs/models/functional_query_encoder.py'), Path('radio_gs/querying/registered_coordinate_query.py')]},
        result_sha256=sha256_file(metadata.output),
        protocol_sha256=sha256_file(metadata.output.with_suffix('.protocol.json')),
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
