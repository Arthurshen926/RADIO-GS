"""Disable lexical takeover globally; preserve the existing native scope path.

This is a rejected diagnostic control, not a deployed replacement. It fixes a
reviewed bag collision but loses substantial ramen text IoU.
"""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', default='native_scope_selection_control')
    args = parser.parse_args()
    parent = args.root / 'counterfactual_extent_benefit_admission'
    receipt = json.loads((parent / 'report.json').read_text())
    assert sha256_file(parent / 'field.pt') == receipt['field_sha256']
    payload = torch.load(parent / 'field.pt', map_location='cpu', weights_only=False)
    state = {key: value.clone() for key, value in payload['state_dict'].items()}
    state['grounded_transfer.acceptance'].fill_(2.)
    output = args.root / args.name
    if output.exists():
        existing = torch.load(output / 'field.pt', map_location='cpu', weights_only=False)
        assert payload['architecture'] == existing['architecture']
        assert state.keys() == existing['state_dict'].keys()
        for key, value in state.items():
            assert torch.equal(value, existing['state_dict'][key]), key
        print('Existing control exactly matches the exported state:', output)
        return
    output.mkdir()
    torch.save(dict(payload, state_dict=state), output / 'field.pt')
    report = dict(schema='native_scope_selection_control.v1',
                  field_sha256=sha256_file(output / 'field.pt'),
                  parent_field_sha256=receipt['field_sha256'],
                  source_only=True, benchmark_data_used=False, training=False,
                  change='Disable all lexical routes by cosine acceptance=2; '
                         'use the existing native-support-overlap path. '
                         'All other tensors unchanged; no query-specific exceptions.',
                  implementation_sha256=sha256_file(__file__))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
