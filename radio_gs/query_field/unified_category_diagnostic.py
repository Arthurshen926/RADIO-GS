"""Fixed existing category readout intervention on sealed pilot predictions.

No training, query re-encoding, parameter search, or new field. This is an
explicit post-pilot development diagnostic, not independent test evidence.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .bounded_experiment import BoundedExperiment
from .category_evidence import corroborated_category_scores
from .storage import sha256


def diagnose(config, study_root, comparison_name, evaluation_path):
    study = Path(study_root)
    barrier = json.loads((study / comparison_name / "prediction_barrier.json").read_text())
    store = BoundedExperiment(study, "category_readout_diagnostic", config["storage"])
    store.json("design.json", dict(parent_comparison=comparison_name,
          rule="existing corroborated_category_scores; temperature=100; positive_support_rank",
          category_source_sha256=sha256(Path(__file__).with_name("category_evidence.py")),
          ranking_source_sha256=sha256(Path(__file__).parents[1] / "querying/observation_readout.py"),
          code_sha256=sha256(__file__), evaluation_sha256=sha256(evaluation_path),
          fixed_before_new_scoring=True, previous_development_metrics_seen=True,
          benchmark_selected_parameters=False, new_model=False))
    data, predictions = {}, {}
    for arm in ("reconstruction", "functional"):
        path = study / f"{comparison_name}_{arm}" / "predictions.pt"
        if sha256(path) != barrier[arm]:
            raise ValueError("parent prediction changed")
        data[arm] = torch.load(path, map_location="cpu", weights_only=True)
        predictions[arm] = corroborated_category_scores(data[arm]["native"], data[arm]["member"])
        store.tensor(arm + ".pt", dict(typed=predictions[arm], parent_predictions_sha256=barrier[arm]),
                     max_bytes=16 * 1024**2)
    store.json("prediction_barrier.json", {a: sha256(store.root / (a + ".pt")) for a in predictions})
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    from radio_gs.scannet_constants import compute_split_metrics
    spec = json.loads(Path(evaluation_path).read_text())
    vertices = PlyData.read(spec["ground_truth"])["vertex"].data
    xyz = np.stack([vertices[k] for k in ("x", "y", "z")], 1)
    truth = np.asarray(vertices["label"])
    results = {}
    for arm, scores in predictions.items():
        parent = data[arm]
        _, indices = cKDTree(parent["xyz"].numpy()).query(xyz, workers=4)
        metrics = {}
        for split, ids in spec["splits"].items():
            columns = [next(i for i, q in enumerate(parent["queries"]) if q["class_id"] == c) for c in ids]
            labels = np.asarray(ids)[scores[:, columns].argmax(1).numpy()]
            labels[~parent["observed"].numpy()] = 0
            metrics[split] = compute_split_metrics(labels[indices], truth, ids)
        results[arm] = metrics
    report = dict(metrics=results, independent_test=False, new_training=False,
                  same_field_and_same_prediction_inputs=True, ground_truth_sha256=sha256(spec["ground_truth"]))
    store.json("results.json", report)
    return {a: {s: m["miou"] for s, m in splits.items()} for a, splits in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--comparison-name", required=True)
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(diagnose(json.loads(Path(args.config).read_text()), args.study_root,
                              args.comparison_name, args.evaluation), indent=2))


if __name__ == "__main__":
    main()
