"""Pixel-scope conflicts from image correspondences independent of the map.

An RGB match can contradict a mask complement, but cannot certify that the
positive proposal has the right identity. We therefore only remove matched
negative supervision; positives, unmatched pixels and native features remain.
"""
from __future__ import annotations

import torch


def camera_neighbor_pairs(world_to_camera: torch.Tensor) -> list[tuple[int, int]]:
    poses = torch.linalg.inv(world_to_camera.double())
    centers, directions = poses[:, :3, 3], poses[:, :3, 2]
    translation = torch.cdist(centers, centers)
    scale = translation[translation > 0].median()
    distance = translation / scale + 1 - directions @ directions.T
    distance.fill_diagonal_(torch.inf)
    neighbors = distance.argsort(dim=1)[:, :2]
    return sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(neighbors) for j in row})


def epipolar_residual(x0, x1, k0, k1, pose0, pose1):
    relative = pose1.double() @ torch.linalg.inv(pose0.double())
    tx, ty, tz = relative[:3, 3]
    skew = torch.stack([torch.stack([tx * 0, -tz, ty]),
                        torch.stack([tz, tx * 0, -tx]),
                        torch.stack([-ty, tx, tx * 0])])
    fundamental = torch.linalg.inv(k1.double()).T @ skew @ relative[:3, :3] @ torch.linalg.inv(k0.double())
    a = torch.cat([x0.double(), torch.ones(len(x0), 1)], dim=1)
    b = torch.cat([x1.double(), torch.ones(len(x1), 1)], dim=1)
    line1, line0 = a @ fundamental.T, b @ fundamental
    error = (line1 * b).sum(1).square()
    return (2 * error / (line1[:, :2].square().sum(1) + line0[:, :2].square().sum(1)).clamp_min(1e-20)).sqrt().float()


def raster_indices(xy: torch.Tensor, image_hw, raster_hw):
    h, w = raster_hw
    x = (xy[:, 0] * w / image_hw[1]).long().clamp(0, w - 1)
    y = (xy[:, 1] * h / image_hw[0]).long().clamp(0, h - 1)
    return y * w + x


def matched_negative_conflicts(pos0, neg0, active0, ids0, pos1, neg1, active1, ids1):
    """Return [pixel, query] unknown additions without propagating positives."""
    active = active0 & active1
    contradiction0 = neg0[ids0] & pos1[ids1] & active[None]
    contradiction1 = neg1[ids1] & pos0[ids0] & active[None]
    out0, out1 = torch.zeros_like(neg0), torch.zeros_like(neg1)
    a, q = torch.where(contradiction0)
    out0[ids0[a], q] = True
    a, q = torch.where(contradiction1)
    out1[ids1[a], q] = True
    return out0, out1
