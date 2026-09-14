"""Persist a source-replayed projection repair without changing evidence tensors."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.evaluation.lerf_object_ceiling import _load_state, _build_carrier
from radio_gs.v4.contracts.build_lerf_object_hypotheses import validate_hypothesis_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-directory", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference_report.read_text())
    report = json.loads((args.repair_directory / "report.json").read_text())
    if report["source_projection_exact"] is not True or report["membership_changed"] is not False:
        raise ValueError("repair lacks exact source replay")
    if report["source_state_sha256"] != reference["scene_state_sha256"]:
        raise ValueError("repair source differs from reference")
    state = _load_state(Path(reference["scene_state"]), expected_sha256=reference["scene_state_sha256"])
    hypothesis_path = Path(reference["hypothesis_memory"])
    if sha256_file(hypothesis_path) != report["hypothesis_memory_sha256"]:
        raise ValueError("repair hypothesis differs")
    hypothesis = torch.load(hypothesis_path, map_location="cpu", weights_only=False)
    if sha256_file(Path(hypothesis["fragment_memory"])) != hypothesis["fragment_memory_sha256"]:
        raise ValueError("fragment digest differs")
    fragment = torch.load(hypothesis["fragment_memory"], map_location="cpu", weights_only=False)
    config = report["carrier_configuration"]
    if list(fragment["raster_shape"]) != list(config["reference_raster_shape"]) or report["source_frame_count"] != len(fragment["source_frames"]):
        raise ValueError("reference scale lacks complete source replay")
    previous = dict(config)
    previous.pop("reference_raster_shape")
    if previous != state["method_configuration"]["carrier"]:
        raise ValueError("repair changed more than reference scale")
    target = args.repair_directory / "sealed"
    target.mkdir(exist_ok=False)
    state.pop("scene_state_sha256", None)
    state["method_configuration"]["carrier"] = config
    state["resolution_repair_receipt"] = {"original_state_sha256": reference["scene_state_sha256"],
        "validation_report_sha256": sha256_file(args.repair_directory / "report.json"), "source_projection_exact": True}
    state_path = (target / "scene_state.pt").resolve()
    torch.save(state, state_path)
    state_sha = sha256_file(state_path)
    fragment.update(scene_state=str(state_path), scene_state_sha256=state_sha)
    fragment_path = (target / "fragment_memory.pt").resolve()
    torch.save(fragment, fragment_path)
    hypothesis.update(scene_state=str(state_path), scene_state_sha256=state_sha,
                      fragment_memory=str(fragment_path), fragment_memory_sha256=sha256_file(fragment_path))
    validate_hypothesis_memory(hypothesis)
    hypothesis_path = (target / "hypothesis_memory.pt").resolve()
    torch.save(hypothesis, hypothesis_path)
    # Reopen the persisted state through the normal evaluator loader.
    loaded = _load_state(state_path, expected_sha256=state_sha)
    assert _build_carrier(loaded).reference_raster_shape == tuple(fragment["raster_shape"])
    (target / "manifest.json").write_text(json.dumps({"scene": reference["scene_label"],
        "scene_state": str(state_path), "scene_state_sha256": state_sha,
        "hypothesis_memory": str(hypothesis_path), "hypothesis_memory_sha256": sha256_file(hypothesis_path),
        "reference_raster_shape": config["reference_raster_shape"], "source_projection_exact": True,
        "method_quality_validated": False}, indent=2) + "\n")
    print(reference["scene_label"], "sealed", flush=True)


if __name__ == "__main__":
    main()
