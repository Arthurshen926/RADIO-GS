"""Replay scalar source-depth transforms used by a frozen mapping carrier."""
import math


def effective_mapping_depth_affine(frame, calibration, recovery, mapping_frames,
                                   mapping_scales, world_units_per_metre):
    """Return ``a, b`` for raw MoGe Z -> frozen-carrier world depth.

    A rejected affine fit is never used: its accepted neighbor recovery starts
    from globally scaled raw depth. The carrier's final mapping scale then acts
    on both depth and offset. Heldout/nonmapping views have no such contract.
    """
    if calibration.get('source_geometry_override') is not None:
        raise ValueError('materialized depth cannot be reduced to scalar affine')
    frames, scales = list(mapping_frames), list(mapping_scales)
    if len(frames) != len(scales) or len(set(frames)) != len(frames):
        raise ValueError('invalid mapping frame/scale binding')
    if frame not in frames:
        raise ValueError('view absent from frozen mapping frames')
    joint = float(scales[frames.index(frame)])
    unit = float(world_units_per_metre)
    if calibration['accepted']:
        local, offset = float(calibration['scale']), float(calibration['offset'])
    elif recovery and recovery.get('accepted', False):
        if recovery.get('frame') != frame:
            raise ValueError('recovery frame identity differs')
        local, offset = float(recovery['scale']), 0.
    else:
        raise ValueError('source has neither accepted affine nor scale recovery')
    if not all(math.isfinite(x) and x > 0 for x in [joint, unit, local]) or not math.isfinite(offset):
        raise ValueError('invalid depth transform')
    return unit * local * joint, offset * joint
