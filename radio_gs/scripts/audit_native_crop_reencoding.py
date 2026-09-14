"""Source-only deterministic crop sentinels; no retrieval ground truth is read."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime
from radio_gs.v4.contracts.build_lerf_fragment_language_memory import _validate_frame_payload, build_frame_memory
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    runtime = None
    reports = []
    for scene in ("figurines", "ramen", "teatime", "waldo_kitchen"):
        manifest_path = args.language_root / scene / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if runtime is None:
            if sha256_file(Path(manifest["radio_checkpoint"])) != manifest["radio_checkpoint_sha256"]:
                raise ValueError("checkpoint hash mismatch")
            runtime = OfficialCropSummaryRuntime.load(checkpoint_path=manifest["radio_checkpoint"], device="cuda:0", parameter_dtype=torch.float16)
        if runtime.radio_checkpoint_sha256 != manifest["radio_checkpoint_sha256"]:
            raise ValueError("scene teacher differs from loaded checkpoint")
        row = manifest["outputs"][0]
        for key in ("source_image", "sam_fragment_payload", "output"):
            if sha256_file(Path(row[key])) != row[key + "_sha256"]:
                raise ValueError("sentinel input hash mismatch")
        payload = torch.load(row["sam_fragment_payload"], map_location="cpu", weights_only=False)
        masks, boxes = _validate_frame_payload(payload, image_path=Path(row["source_image"]))
        # First three proposals by cache order, selected without labels or queries.
        count = min(3, len(masks))
        for key in ("quality", "stability", "parent_index", "proposal_area_fraction"):
            payload[key] = payload[key][:count]
        record = {"frame_id": row["frame_id"], "image_path": Path(row["source_image"]),
                  "image_sha256": row["source_image_sha256"], "sam_path": Path(row["sam_fragment_payload"]),
                  "sam_sha256": row["sam_fragment_payload_sha256"], "payload": payload, "masks": masks[:count], "boxes": boxes[:count]}
        contract = manifest["descriptor_contract"]
        fresh = build_frame_memory(record, runtime, device=torch.device("cuda:0"), crop_resolution=contract["crop_resolution"],
                                   batch_size=1, context_expansion=contract["context_expansion"], masked_background_rgb=tuple(contract["masked_background_rgb"]))
        cached = torch.load(row["output"], map_location="cpu", weights_only=False)
        result = {"scene": scene, "frame_id": row["frame_id"], "proposal_count": count, "manifest_sha256": sha256_file(manifest_path)}
        for key in ("masked_crop_descriptor", "context_crop_descriptor"):
            a, b = fresh[key].float().cpu(), cached[key][:count].float().cpu()
            result[key] = {"max_absolute_error": float((a-b).abs().max()),
                           "minimum_cosine": float(torch.nn.functional.cosine_similarity(a,b).min())}
        reports.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump({"source_only": True, "text_queries_opened": False, "retrieval_accuracy_measured": False,
                   "summary_slot_index": runtime.summary_slot_index, "results": reports,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}, output, indent=2)


if __name__ == "__main__":
    main()
