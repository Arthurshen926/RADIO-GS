"""Verify actual protected tensors and teacher exposure in the expanded rebuild."""
import json
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    torch.set_num_threads(4)
    root = Path('output/scannet_expanded_budget_next29')
    plan = json.loads((root/'adaptive_plan.json').read_text())
    rows = {}
    for scene in plan['rebuild_scenes']:
        field = root/scene/'base/field'
        teachers = json.loads((field/'teachers.json').read_text())
        receipt = json.loads((field/'ropefix96_protected/report.json').read_text())
        initialization = field/'source_fusion_initialization.pt'
        initialization_receipt = json.loads(initialization.with_suffix('.json').read_text())
        assert sha256_file(initialization) == initialization_receipt['sha256']
        checkpoint = field/'ropefix96_protected/field.pt'
        assert sha256_file(checkpoint) == receipt['field_sha256']
        initial = torch.load(initialization, map_location='cpu', weights_only=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
        valid = initial['valid']
        dim = len(initial['basis'])
        assert torch.equal(state['observed'], valid)
        assert torch.equal(state['latent'][valid, :dim], initial['latent'][valid])
        assert torch.equal(state['native_decoder.weight'], initial['basis'].T)
        assert torch.count_nonzero(state['native_decoder.bias']) == 0
        coverage = receipt['teacher_pixel_coverage']
        assert len(coverage) == len(teachers['frames'])
        train = set(teachers['train_frames'])
        validation = set(teachers['validation_frames'])
        assert len(train) == 32 and not train & validation
        assert set(teachers['frames']) == train | validation
        for frame, count in zip(teachers['frames'], coverage):
            if frame in train:
                assert count['valid'] > 0 and count['visited'] == count['valid']
                assert receipt['dense_teacher_view_updates'][str(frame)] > 0
            else:
                assert count['visited'] == 0
                assert receipt['dense_teacher_view_updates'].get(str(frame), 0) == 0
        rows[scene] = dict(initialization_sha256=sha256_file(initialization),
                           field_sha256=sha256_file(checkpoint),
                           report_sha256=sha256_file(field/'ropefix96_protected/report.json'),
                           observed_elements=int(valid.sum()), protected_native_exact=True,
                           all_training_teacher_pixels_visited=True,
                           validation_teacher_pixels_visited=0)
    result = dict(complete=True, source_only=True, independent_test=False,
                  scope='Base training protection; subsequent pixel fusion deliberately rewrites the native prefix.',
                  rows=rows, implementation_sha256=sha256_file(__file__))
    (root/'native_protection_audit.json').open('x').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
