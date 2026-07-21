"""Query-view selection exports.

The implementation is centralized in :mod:`protocol` so the CLI, tests and
release freezer cannot silently diverge.
"""

from radio_gs.benchmarks.scannet_pfir.protocol import (
    FrameInstanceObservation,
    FramePaths,
    _choose_two_views as choose_easy_and_hard_views,
    exclusion_frame_ids,
)

__all__ = [
    "FrameInstanceObservation",
    "FramePaths",
    "choose_easy_and_hard_views",
    "exclusion_frame_ids",
]

