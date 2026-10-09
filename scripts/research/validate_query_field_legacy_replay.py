"""Capture the original hash check's exact GPU operation sequence and buffers."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.research import validate_query_field_mixture_replay as audit
from scripts.research.validate_query_field_program import results as original_results


def capture(directory, fixture):
    count = audit.read(Path(directory) / 'manifest.json')['elements']
    original_sha = hashlib.sha256; buffers = []
    def capture_sha(data=b'', *args, **kwargs):
        if len(data) in {count, 4*count, 8*count, 4*count*len(fixture['category_queries'])}:
            buffers.append(bytes(data))
        return original_sha(data, *args, **kwargs)
    # No extra CUDA calls, retained GPU tensors or changed query sequence. The
    # original reader has already copied these byte buffers to the CPU to hash.
    hashlib.sha256 = capture_sha
    try:
        hashes = original_results(directory, fixture, 'common_program')
    finally:
        hashlib.sha256 = original_sha
    if len(buffers) != 10: raise ValueError('unexpected original hash sequence')
    outputs = {}
    for index, name in enumerate(['text', 'image', 'point_2d', 'point_3d', 'category']):
        scores, selected = buffers[2*index:2*index+2]
        if original_sha(scores).hexdigest() != hashes[name]['scores_sha256'] or original_sha(selected).hexdigest() != hashes[name]['selected_sha256']:
            raise ValueError('captured buffers do not match the original audit')
        outputs[name] = dict(scores=torch.from_numpy(np.frombuffer(scores, np.float32).copy()),
            selected=torch.from_numpy(np.frombuffer(selected, np.int64 if name == 'category' else np.bool_).copy()))
    return outputs


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--child', action='store_true')
    p.add_argument('--field'); p.add_argument('--fixture'); p.add_argument('--output'); p.add_argument('--deny', action='append', default=[])
    args = p.parse_args(); torch.set_num_threads(4)
    audit.results = capture
    audit.CHILD_MODULE = 'scripts.research.validate_query_field_legacy_replay'
    audit.VALIDATION_DIRECTORY = 'runtime_legacy_numeric'
    audit.child(args) if args.child else audit.main()
