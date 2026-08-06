"""Deterministic source-reference completion helpers for NVOS."""

from __future__ import annotations

import numpy as np


SPLITMIX_SALT = 0x243F6A8885A308D3
UINT64_MASK = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK


def deterministic_positive_points(
    mask: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    """Select signed-registration points from a 2D binary mask."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("positive scribble must be a 2D mask")
    if int(count) <= 0:
        raise ValueError("point count must be positive")
    height, width = values.shape
    pixel_ids = np.flatnonzero(values.reshape(-1))
    if pixel_ids.size < int(count):
        raise ValueError(
            f"positive scribble has {pixel_ids.size} pixels but {count} are required"
        )
    ordered = sorted(
        (int(pixel_id) for pixel_id in pixel_ids),
        key=lambda pixel_id: (splitmix64(pixel_id ^ SPLITMIX_SALT), pixel_id),
    )[: int(count)]
    return np.asarray(
        [(pixel_id % width, pixel_id // width) for pixel_id in ordered],
        dtype=np.float32,
    )


def aggregate_completed_positive(
    trial_masks: np.ndarray,
    raw_positive: np.ndarray,
    raw_negative: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average binary trials and apply the frozen scribble overwrite policy."""

    trials = np.asarray(trial_masks, dtype=bool)
    positive = np.asarray(raw_positive, dtype=bool)
    negative = np.asarray(raw_negative, dtype=bool)
    if trials.ndim != 3 or trials.shape[1:] != positive.shape:
        raise ValueError("trial masks and positive scribble do not align")
    if negative.shape != positive.shape:
        raise ValueError("positive and negative scribbles do not align")
    if not np.isfinite(float(threshold)):
        raise ValueError("completion threshold must be finite")
    aggregate = trials.astype(np.float32).mean(axis=0, dtype=np.float32)
    completed = aggregate >= float(threshold)
    completed[positive] = True
    completed[negative] = False
    return aggregate, completed


def entropy_reliability_soft_observation(
    aggregate_probability: np.ndarray,
    raw_positive: np.ndarray,
    raw_negative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the registered parameter-free binary-entropy reliability."""

    values = np.asarray(aggregate_probability, dtype=np.float64)
    positive = np.asarray(raw_positive, dtype=bool)
    negative = np.asarray(raw_negative, dtype=bool)
    if (
        values.ndim != 2
        or positive.shape != values.shape
        or negative.shape != values.shape
    ):
        raise ValueError("aggregate and signed scribbles must be aligned 2D arrays")
    if not np.isfinite(values).all() or bool(((values < 0) | (values > 1)).any()):
        raise ValueError("aggregate probability must be finite in [0,1]")
    if bool(np.logical_and(positive, negative).any()):
        raise ValueError("raw positive and negative scribbles overlap")
    reliability = np.ones_like(values, dtype=np.float64)
    interior = (values > 0) & (values < 1)
    probability = values[interior]
    entropy = -probability * np.log(probability) - (
        1.0 - probability
    ) * np.log1p(-probability)
    reliability[interior] = 1.0 - entropy / np.log(2.0)
    reliability = np.clip(reliability, 0.0, 1.0)
    observation = values * reliability
    observation[positive] = 1.0
    observation[negative] = 0.0
    return (
        reliability.astype(np.float32, copy=False),
        observation.astype(np.float32, copy=False),
    )


def probability_preserving_entropy_observation(
    aggregate_probability: np.ndarray,
    raw_positive: np.ndarray,
    raw_negative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep source probability and reliability as separate Bernoulli terms.

    The historical soft-completion diagnostic returned ``q * c(q)`` as if it
    were a foreground probability.  That turns low confidence into negative
    evidence.  This helper instead returns the overwritten foreground
    probability ``q`` and the independent entropy reliability ``c(q)``.  A
    downstream exact adjoint can therefore pool ``c*q`` and ``c*(1-q)`` and
    recover both the conditional foreground probability and its confidence.
    """

    values = np.asarray(aggregate_probability, dtype=np.float64)
    positive = np.asarray(raw_positive, dtype=bool)
    negative = np.asarray(raw_negative, dtype=bool)
    if values.ndim != 2 or positive.shape != values.shape or negative.shape != values.shape:
        raise ValueError("aggregate and signed scribbles must be aligned 2D arrays")
    if not np.isfinite(values).all() or bool(((values < 0) | (values > 1)).any()):
        raise ValueError("aggregate probability must be finite in [0,1]")
    if bool(np.logical_and(positive, negative).any()):
        raise ValueError("raw positive and negative scribbles overlap")

    reliability = np.ones_like(values, dtype=np.float64)
    interior = (values > 0) & (values < 1)
    probability = values[interior]
    entropy = -probability * np.log(probability) - (
        1.0 - probability
    ) * np.log1p(-probability)
    reliability[interior] = 1.0 - entropy / np.log(2.0)
    reliability = np.clip(reliability, 0.0, 1.0)
    probability = values.copy()
    probability[positive] = 1.0
    probability[negative] = 0.0
    reliability[positive | negative] = 1.0
    return (
        probability.astype(np.float32, copy=False),
        reliability.astype(np.float32, copy=False),
    )
