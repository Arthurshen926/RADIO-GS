"""Native region retrieval for runtime texts lacking direct detector evidence.

Selection uses source features and the current encoding only. Top1 is a seed
hypothesis, not independently confirmed identity or a complete category set.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--response', required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    scene = json.loads(Path(a.scene).read_text())
    response = load(a.response)
    provider = json.loads((Path(a.evidence)/'provider.json').read_text())
    ids = [i for i,t in enumerate(response['texts']) if provider['detected_views'][t] == 0]
    query = F.normalize(response['encodings'][ids].float(), dim=-1)
    winners = {i:None for i in ids}
    for directory in scene['native_region_directories']:
        for frame in scene['source_frames']:
            path = Path(directory)/f'frame_{frame:05d}.pt'
            native = load(path)
            if not len(native['descriptors']):
                continue
            scores = F.normalize(native['descriptors'].float(), dim=-1) @ query.T
            values, indices = scores.max(0)
            for j, qi in enumerate(ids):
                candidate = dict(path=str(path), frame=frame, index=int(indices[j]), score=float(values[j]))
                if winners[qi] is None or candidate['score'] > winners[qi]['score']:
                    winners[qi] = candidate
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    seeds = {}
    for qi, winner in winners.items():
        native = load(winner['path'])
        sam = load(native['sam_path'])
        masks = unpack_masks(sam['packed_masks'], sam['mask_shape'][1])
        mask = masks[int(native['native_proposal_indices'][winner['index']])].copy()
        seed = dict(query=response['texts'][qi], query_index=qi, seed_frame=winner['frame'],
                    seed_confidence=max(0.,winner['score']), masks=torch.from_numpy(mask)[None].bool(),
                    source='current text native top1 region; unverified identity hypothesis', candidate=winner)
        path = out/f'{qi}.pt'
        torch.save(seed,path)
        seeds[str(qi)] = str(path)
    (out/'seeds.json').write_text(json.dumps(dict(seeds=seeds, query_indices=ids, candidates=winners,
        policy='only zero-direct-detection queries; native top1 over all declared source region banks',
        geometry_or_pair_rank_filter=False, historical_query_answers=False),indent=2)+'\n')
    print(scene['scene'],len(seeds),'fresh seeds',flush=True)


if __name__ == '__main__':
    main()
