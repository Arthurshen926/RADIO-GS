#!/usr/bin/env python3
"""Nested-source calibration rescue for DINO physical-track association."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from radio_gs.scripts.train_lerf_source_physical_track_calibrator import rank_auc
from radio_gs.utils.immutable_artifacts import sha256_file


SCHEMA = "radio_gs.lerf_source_physical_track_calibrator_nested.v1"
TEMPERATURE_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
JEFFREYS_STRENGTH_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def jeffreys_probability(logit: torch.Tensor, temperature: float, strength: float) -> torch.Tensor:
    """Apply temperature then a symmetric Jeffreys Beta pseudo-count."""

    probability = torch.sigmoid(logit / float(temperature))
    alpha = float(strength)
    return (probability + 0.5 * alpha) / (1.0 + alpha)


def balanced_probability_log_score(probability: torch.Tensor, label: torch.Tensor) -> float:
    p = probability.clamp(1e-7, 1 - 1e-7)
    loss = -(label.float() * p.log() + (1 - label.float()) * (1 - p).log())
    return float(0.5 * loss[label == 1].mean() + 0.5 * loss[label == 0].mean())


def fit_logistic(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-4)
    weight = torch.zeros(x.shape[1], requires_grad=True); bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad(); logit = ((x - mean) / scale) @ weight + bias
        raw = F.binary_cross_entropy_with_logits(logit, y.float(), reduction="none")
        loss = 0.5 * raw[y == 1].mean() + 0.5 * raw[y == 0].mean() + 1e-3 * weight.square().sum()
        loss.backward(); return loss
    optimizer.step(closure)
    return mean, scale, weight.detach(), bias.detach()


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "radio_gs.lerf_source_dino_physical_track_authority.v1":
        raise ValueError("authority schema differs")
    if payload.get("metadata", {}).get("figurines_opened") is not False:
        raise ValueError("authority opened figurines")
    return payload


def build(args: argparse.Namespace) -> dict:
    assets = []
    inner_train_x, inner_train_y = [], []
    outer_train_x, outer_train_y = [], []
    inner_eval = []; outer_eval = []
    feature_names = None
    for value in args.authority:
        path = Path(value).resolve(); payload = _load(path)
        names = list(payload["feature_names"])
        if feature_names is None: feature_names = names
        if names != feature_names: raise ValueError("feature schemas differ")
        x = torch.as_tensor(payload["edge_features"]).float(); y = torch.as_tensor(payload["edge_label"]).long()
        left = torch.as_tensor(payload["edge_left"]).long(); right = torch.as_tensor(payload["edge_right"]).long()
        views = torch.as_tensor(payload["proposal_views"]).long(); known = y >= 0
        outer_hold = known & ((views[left] % 4 == 3) | (views[right] % 4 == 3))
        outer_train = known & ~outer_hold
        inner_hold = outer_train & ((views[left] % 4 == 2) | (views[right] % 4 == 2))
        inner_train = outer_train & ~inner_hold
        for split in (inner_train, inner_hold, outer_hold):
            labels = y[split]
            if not bool((labels == 0).any() and (labels == 1).any()):
                raise ValueError(f"{payload['scene']} nested split lacks both classes")
        inner_train_x.append(x[inner_train]); inner_train_y.append(y[inner_train])
        outer_train_x.append(x[outer_train]); outer_train_y.append(y[outer_train])
        inner_eval.append((str(payload["scene"]), x[inner_hold], y[inner_hold]))
        outer_eval.append((str(payload["scene"]), x[outer_hold], y[outer_hold]))
        assets.append({"scene": payload["scene"], "path": str(path), "sha256": sha256_file(path)})

    inner_x, inner_y = torch.cat(inner_train_x), torch.cat(inner_train_y)
    i_mean, i_scale, i_weight, i_bias = fit_logistic(inner_x, inner_y)
    candidates = []
    for temperature in TEMPERATURE_GRID:
        for strength in JEFFREYS_STRENGTH_GRID:
            scene_scores = []
            for _, x, y in inner_eval:
                logit = ((x - i_mean) / i_scale) @ i_weight + i_bias
                scene_scores.append(balanced_probability_log_score(jeffreys_probability(logit, temperature, strength), y))
            candidates.append((sum(scene_scores) / len(scene_scores), temperature, strength, scene_scores))
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    nested_score, temperature, strength, nested_scene_scores = candidates[0]

    outer_x, outer_y = torch.cat(outer_train_x), torch.cat(outer_train_y)
    mean, scale, weight, bias = fit_logistic(outer_x, outer_y)
    reports = {}; passed = True
    with torch.no_grad():
        for scene, x, y in outer_eval:
            logit = ((x - mean) / scale) @ weight + bias
            probability = jeffreys_probability(logit, temperature, strength)
            proper = balanced_probability_log_score(probability, y); auc = rank_auc(logit, y)
            scene_pass = auc > 0.5 and proper < 0.6931471805599453; passed &= scene_pass
            reports[scene] = {"known_edges": int(y.numel()), "same_edges": int((y == 1).sum()),
                              "different_edges": int((y == 0).sum()), "auc": auc,
                              "balanced_log_score": proper, "epoch0_balanced_log_score": 0.6931471805599453,
                              "pass": scene_pass}
    output = Path(args.output).resolve(); report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists(): raise FileExistsError(f"output exists: {output}")
    payload = {"schema": SCHEMA, "schema_version": 1, "feature_names": feature_names,
               "feature_mean": mean, "feature_scale": scale, "weight": weight, "bias": bias,
               "temperature": temperature, "jeffreys_strength": strength,
               "metadata": {"source_only": True, "figurines_opened": False,
                            "outer_holdout": "edge_touches_view_mod4_eq3",
                            "inner_holdout": "within_outer_train_edge_touches_view_mod4_eq2",
                            "selection": "minimum_macro_scene_inner_balanced_log_score_fixed_grid",
                            "temperature_grid": list(TEMPERATURE_GRID),
                            "jeffreys_strength_grid": list(JEFFREYS_STRENGTH_GRID), "source_assets": assets}}
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, output)
    report = {"schema": SCHEMA, "status": "source_heldout_gate_pass" if passed else "source_heldout_gate_fail",
              "formal_stage_a_complete": passed, "figurines_opened": False,
              "nested_selection": {"temperature": temperature, "jeffreys_strength": strength,
                                   "macro_balanced_log_score": nested_score,
                                   "per_scene_balanced_log_score": dict(zip([x[0] for x in inner_eval], nested_scene_scores))},
              "scenes": reports, "output": str(output), "output_sha256": sha256_file(output)}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--authority", nargs="+", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__": main()
