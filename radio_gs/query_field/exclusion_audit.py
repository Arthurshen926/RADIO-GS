"""Source-only audit of known exclusions, before any supervision intervention."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .contracts import Support
from .geometry import selected_alpha
from .observations import erode, partial_mask_target, training_episode
from .storage import load_field, sha256


def disjoint_target(masks, index):
    """No target-overlapping mask can supply an exclusion, regardless of size."""
    overlap = (masks.flatten(1) & masks[index].flatten()).any(1)
    negative = erode(masks[~overlap]).any(0) if bool((~overlap).any()) else torch.zeros_like(masks[index])
    return negative & ~masks[index]


@torch.no_grad()
def run(config, output, device='cuda'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('preserve completed source audit')
    root = Path(config['observations'])
    observations = json.loads((root/'observations.json').read_text())
    field, manifest = load_field(config['field'], device)
    if manifest['provenance']['observations_sha256'] != sha256(root/'observations.json'):
        raise ValueError('field and source observation construction differ')
    views, episodes, rows = [], [], []
    observed = field.observed.cpu()
    for entry in observations['frames']:
        if entry['split'] != 'train':
            continue
        path = root/f"view_{entry['frame']:05d}.pt"
        if sha256(path) != entry['compiled']['sha256']:
            raise ValueError('source observations changed')
        view = torch.load(path, map_location='cpu', weights_only=True)
        views.append(view)
        visible = view['raw_mass'].reshape(view['masks'].shape[-2:]) > 0
        for index in range(len(view['masks'])):
            episode = training_episode(view, index, observed)
            if episode is None:
                continue
            positive, negative, pixels = episode
            extended = (disjoint_target(view['masks'], index) & visible).flatten()
            assert not (negative & ~extended).any() and not (positive & extended).any()
            row = dict(frame=entry['frame'], region=index, positive_pixels=int(positive.sum()),
                       comparable_negative_pixels=int(negative.sum()), added_negative_pixels=int((extended & ~negative).sum()))
            rows.append(row)
            episodes.append((len(views)-1, index, pixels, negative, extended))
    positions = torch.linspace(0, len(episodes)-1, min(config['rendered_episodes'], len(episodes))).round().long().tolist()
    rendered = []
    height, width = observations['raster_hw']
    for position in positions:
        vi, index, pixels, negative, extended = episodes[position]
        view = views[vi]
        anchor = view['best_element'][pixels[len(pixels)//2]].to(device)
        scale = float(view['scales_m'][index])
        member = field.member_logits(Support(anchor.reshape(1), torch.ones(1, device=device)), scale).sigmoid()
        selection = (member > field.threshold) & field.observed
        camera = SimpleNamespace(intrinsic=view['intrinsic'], world_to_camera=view['world_to_camera'],
                                 height=view['original_hw'][0], width=view['original_hw'][1])
        mask = selected_alpha(field, selection, camera, height, width, observations['world_units_per_metre']).cpu() > .5
        target = view['masks'][index].flatten()
        false_additions = mask & ~target
        denominator = max(1, int(false_additions.sum()))
        rendered.append(dict(frame=view['frame'], region=index, false_addition_pixels=int(false_additions.sum()),
            comparable_negative_false_fraction=int((false_additions & negative).sum())/denominator,
            added_negative_false_fraction=int((false_additions & extended & ~negative).sum())/denominator,
            remaining_unknown_false_fraction=int((false_additions & ~extended).sum())/denominator))
    summary = dict(eligible_episodes=len(rows), rendered_episodes=len(rendered),
        mean_comparable_negative_pixels=sum(r['comparable_negative_pixels'] for r in rows)/len(rows),
        mean_added_negative_pixels=sum(r['added_negative_pixels'] for r in rows)/len(rows),
        mean_added_false_fraction=sum(r['added_negative_false_fraction'] for r in rendered)/len(rendered))
    result = dict(config=config, observations_sha256=sha256(root/'observations.json'),
        field_sha256=manifest['files']['field.pt']['sha256'], source_train_only=True,
        benchmark_labels_opened=False, summary=summary, all_episodes=rows, rendered_episodes=rendered,
        limitation='Source SAM region extent and privileged source scale; not complete-object truth or benchmark accuracy.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(run(json.loads(Path(args.config).read_text()), args.output, args.device), indent=2))
