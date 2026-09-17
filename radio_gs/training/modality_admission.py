"""Explicitly admit missing direct text supervision without inventing labels."""


def admitted_training_modalities(counts, *, allow_missing_text=False):
    if set(counts) != {'text', 'image', 'prompt'} or any(n < 0 for n in counts.values()):
        raise ValueError('three nonnegative modality counts required')
    missing = [m for m in ('text', 'image', 'prompt') if counts[m] == 0]
    if missing and not (allow_missing_text and missing == ['text']):
        raise ValueError('missing direct query supervision: ' + ', '.join(missing))
    return [m for m in ('text', 'image', 'prompt') if counts[m] > 0]
