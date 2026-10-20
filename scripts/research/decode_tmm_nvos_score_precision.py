"""Observe raw logits without editing the frozen N16 decoder or its outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from scripts.research import decode_tmm_nvos_target_redecode as frozen


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True)
    root = Path(parser.parse_args().request).resolve()
    if root.parent != Path('/dev/shm') or root.stat().st_mode & 0o077:
        raise ValueError('private tmpfs required')
    records = []
    original_loader = frozen._load_sam3_model

    def capture(kind, logits):
        array = np.asarray(logits, dtype=np.float32).squeeze()
        assert array.ndim == 2 and np.isfinite(array).all()
        name = f'raw_{kind}_{len(records)}.npy'
        np.save(root / name, array, allow_pickle=False)
        records.append(dict(kind=kind, file=name, shape=list(array.shape)))
        assert sum(p.stat().st_size for p in root.iterdir()) < 128 * 2**20

    def observed_loader(**kwargs):
        sam = original_loader(**kwargs)
        original_predict = sam.model.predict_inst
        def predict(*args, **kw):
            result = original_predict(*args, **kw)
            assert kw['return_logits']
            capture('image', result[0])
            return result
        sam.model.predict_inst = predict
        tracker = sam.model.inst_interactive_predictor.model
        original_propagate = tracker.propagate_in_video
        def propagate(*args, **kw):
            for result in original_propagate(*args, **kw):
                index, ids, low, masks, presence = result
                if int(index) == 1:
                    capture('memory', masks[list(ids).index(1)].float().cpu().numpy())
                yield result
        tracker.propagate_in_video = propagate
        return sam

    frozen._load_sam3_model = observed_loader
    try:
        frozen.main()
    finally:
        frozen._load_sam3_model = original_loader
    (root/'raw_records.json').write_text(json.dumps(records)+'\n')


if __name__ == '__main__': main()
