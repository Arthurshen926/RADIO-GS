"""Cold-check formal native fragment artifacts against the evaluated source fields."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.v4.contracts.lerf_fragment_surface_memory import validate_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    records = []
    for scene in ("figurines", "ramen", "teatime", "waldo_kitchen"):
        path = args.result_root / "v4_native_fragment_repair_20260914" / (scene + ".pt")
        memory = validate_memory(torch.load(path, map_location="cpu", weights_only=False))
        scene_path = Path(memory["scene_state"])
        if sha256_file(scene_path) != memory["scene_state_sha256"] or memory["source_mask_raster"] != "native":
            raise ValueError("formal native memory scene binding differs")
        evaluated = args.result_root / "v4_formal_native_evaluation_v2_20260914" / scene
        ref = json.loads((evaluated / "report.json").read_text())
        if sha256_file(path) != ref["formal_fragment_memory_sha256"]:
            raise ValueError("evaluated formal fragment memory hash differs")
        field_path = evaluated / "source_fields.pt"
        if sha256_file(field_path) != ref["source_contract"]["source_fields_sha256"]:
            raise ValueError("evaluated field hash differs")
        fields = torch.load(field_path, map_location="cpu", weights_only=False)
        comparisons = {}
        for mode, key in (("source_masked_top1", "masked_ids"), ("source_dual_top1", "dual_ids")):
            actual = memory["fragment_positive"][fields[key]].float().T
            expected = fields["fields"][mode]
            comparisons[mode] = {"bit_exact": torch.equal(actual, expected), "max_absolute_error": float((actual - expected).abs().max())}
            if not comparisons[mode]["bit_exact"]:
                raise ValueError(f"{scene} {mode}: regenerated field requires a new evaluation: {comparisons[mode]}")
        records.append({"scene": scene, "fragment_memory": str(path.resolve()), "sha256": sha256_file(path),
                        "scene_state_sha256": memory["scene_state_sha256"], "source_frame_count": len(memory["source_frames"]),
                        "comparison": comparisons})
        print(scene, "formal native fields match evaluated fields exactly", flush=True)
    with args.output.open("x") as file:
        json.dump({"cold_verification": True, "prediction_fields_bit_exact": True, "records": records}, file, indent=2)


if __name__ == "__main__":
    main()
