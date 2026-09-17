"""Choose additional source views by geometry alone, with a sealed exclusion set."""
import torch


def select_coverage_views(visibility, initial_frames, excluded_frames, budget):
    if budget < 0 or len(set(initial_frames)) != len(initial_frames):
        raise ValueError('invalid selection budget or initial frames')
    excluded = set(excluded_frames)
    if excluded.intersection(initial_frames) or not set(initial_frames) <= visibility.keys():
        raise ValueError('initial frames must be available and outside excluded set')
    legal = {f: v.bool() for f, v in visibility.items() if f not in excluded}
    if not legal or len({tuple(v.shape) for v in legal.values()}) != 1 or any(v.ndim != 1 for v in legal.values()):
        raise ValueError('visibility must have a common element axis')
    covered = torch.zeros_like(next(iter(legal.values())))
    for frame in initial_frames:
        covered |= legal[frame]
    initial_count = int(covered.sum())
    remaining = set(legal) - set(initial_frames)
    selected = []
    for _ in range(budget):
        if not remaining:
            break
        frame = min(remaining, key=lambda f: (-int((legal[f] & ~covered).sum()), f))
        gain = int((legal[frame] & ~covered).sum())
        if gain == 0:
            break
        covered |= legal[frame]
        remaining.remove(frame)
        selected.append({'frame': frame, 'additional_elements': gain, 'covered_elements': int(covered.sum())})
    return {'initial_covered_elements': initial_count, 'selection': selected,
            'final_covered_elements': int(covered.sum())}
