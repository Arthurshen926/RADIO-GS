"""Export prespecified semantic-preserving extent controls, without label access."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--root', type=Path)
    p.add_argument('--parent',default='exact_render_query_threshold96')
    p.add_argument('--prefix',default='unified_bounded_extent')
    a = p.parse_args()
    root = a.root or Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916') / a.scene
    parent = root / a.parent
    receipt = json.loads((parent / 'report.json').read_text())
    assert sha256_file(parent / 'field.pt') == receipt['field_sha256']
    payload = torch.load(parent / 'field.pt', weights_only=False)
    for strength in (0., 2.):
        name = a.prefix + str(int(strength))
        out = root / name
        out.mkdir(exist_ok=False)
        arch = dict(payload['architecture'], bounded_extent={'text': strength, 'image': strength})
        field = ElementRetrievalField(**arch)
        field.load_state_dict(payload['state_dict'], strict=True)
        for key, value in field.state_dict().items():
            assert torch.equal(value, payload['state_dict'][key]), key
        torch.save(dict(payload, architecture=arch), out / 'field.pt')
        (out / 'report.json').write_text(json.dumps(dict(
            field_sha256=sha256_file(out / 'field.pt'), parent_field_sha256=receipt['field_sha256'],
            source_only=True, benchmark_data_used=False, strength=strength,
            decision='Prespecified zero and shared image-source-selected strength controls, not benchmark-selected deployment.',
            all_parent_tensors_exact=True, native_semantics_preserved=True,
            guarantee='Native element logits change by at most strength; strong margins protected, not all pairwise rankings.',
            implementation_sources={p: sha256_file(p) for p in [__file__, 'radio_gs/models/element_retrieval_field.py', 'radio_gs/models/bounded_extent_readout.py']},
        ), indent=2) + '\n')


if __name__ == '__main__':
    main()
