"""Independent-process verification of repaired source diagnostic fields."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
results = []
for scene in ('figurines','ramen','teatime','waldo_kitchen'):
    root = args.root/scene
    report = json.loads((root/'report.json').read_text())
    if sha256_file(root/'source_fields.pt') != report['source_contract']['source_fields_sha256']:
        raise ValueError('source field checksum changed')
    if sha256_file(root/'native_relift.pt') != report['native_relift_contract']['payload_sha256']:
        raise ValueError('source evidence checksum changed')
    fields = torch.load(root/'source_fields.pt', map_location='cpu')
    source = torch.load(root/'native_relift.pt', map_location='cpu')
    assert json.dumps(source['carrier_configuration'], sort_keys=True) == json.dumps(report['carrier_configuration'], sort_keys=True)
    for kind in ('masked','dual'):
        rebuilt = source['fragment_positive'][fields[kind+'_ids']].float().T
        assert torch.equal(rebuilt, fields['fields']['source_'+kind+'_top1'])
    assert source['view_visibility'].shape[1] == source['fragment_positive'].shape[1]
    results.append({'scene':scene,'field_rebuilt_from_saved_source_evidence_bit_exact':True,
                    'projection_contract':report['carrier_configuration'].get('projection_contract'),
                    'element_count':source['fragment_positive'].shape[1],
                    'role':'diagnostic_source_field_not_sealed_deployment'})
(args.root/'cold_verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
