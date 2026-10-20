"""Plan reference-to-target memory paths without target labels or task rules."""
def reference_memory_paths(source_frames, reference, target):
    frames=list(source_frames)
    if len(frames)!=len(set(frames)) or reference not in frames or target in frames:
        raise ValueError('unique source frames must contain the reference and exclude the target')
    if any(type(v) is not int for v in [*frames,reference,target]):
        raise ValueError('integer canonical frame indices required')
    lo,hi=sorted([reference,target])
    sequence=sorted([v for v in frames if lo<=v<=hi]+[target],reverse=target<reference)
    if sequence[0]!=reference or sequence[-1]!=target:
        raise ValueError('invalid reference-to-target path')
    return dict(pair_memory=[reference,target],sequence_memory=sequence)
