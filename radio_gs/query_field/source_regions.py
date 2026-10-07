"""Encode authorized query-free SAM regions with the existing native recipe.

This construction-only bridge accepts explicit RGB/cache pairs. It does not
accept benchmark prompts, masks, source tracking, or fitted legacy features.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .storage import sha256
from .observations import source_sam_metadata


@torch.inference_mode()
def encode(config_path, device="cuda"):
    from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
    from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
    config = json.loads(Path(config_path).read_text())
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    runtime = NativeSiglip2Runtime.load(Path(config["encoder"]), device=torch.device(device))
    records = []
    for entry in config["frames"]:
        if sha256(entry["rgb"]) != entry["rgb_sha256"] or sha256(entry["sam"]) != entry["sam_sha256"]:
            raise ValueError("construction input digest differs")
        sam = torch.load(entry["sam"], map_location="cpu", weights_only=False)
        metadata = sam["metadata"]
        source_hash, _ = source_sam_metadata(metadata)
        if source_hash != entry["rgb_sha256"]:
            raise ValueError("source mask/RGB mismatch")
        masks = unpack_masks(sam["packed_masks"], sam["mask_shape"][1])
        rgb = np.array(Image.open(entry["rgb"]).convert("RGB"))
        h, w = rgb.shape[:2]
        if masks.shape[1:] != (h, w):
            raise ValueError("native crop recipe requires masks at RGB resolution")
        crops, ids = [], []
        for index, mask in enumerate(masks):
            if not .002 <= float(mask.mean()) <= .6:
                continue
            yy, xx = np.where(mask)
            x0, x1 = max(0, int(xx.min())-3), min(w, int(xx.max())+4)
            y0, y1 = max(0, int(yy.min())-3), min(h, int(yy.max())+4)
            crop = rgb[y0:y1, x0:x1].copy()
            crop[~mask[y0:y1, x0:x1]] = 127
            crop = Image.fromarray(crop).resize((384, 384), Image.Resampling.BICUBIC)
            crops.append(torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).float()/255)
            ids.append(index)
        if not crops:
            raise ValueError("source frame has no eligible native regions")
        descriptors = F.normalize(torch.cat([runtime.encode(torch.stack(crops[i:i+8])).float().cpu()
                                             for i in range(0, len(crops), 8)]), dim=-1)
        target = out / f"frame_{entry['frame']:05d}.pt"
        torch.save(dict(frame=entry["frame"], descriptors=descriptors.half(),
                        native_proposal_indices=torch.tensor(ids, dtype=torch.long),
                        sam_path=entry["sam"], native_mask_hw=[h, w],
                        encoder=dict(path=config["encoder"]),
                        crop="masked RGB bbox+3pixels, gray127 background,384",
                        query_independent=True), target)
        records.append(dict(**entry, regions=len(ids), output_sha256=sha256(target)))
        print("encoded", entry["frame"], len(ids), flush=True)
    report = dict(config_sha256=sha256(config_path), frames=records,
                  recipe="same native masked crop eligibility and encoder as LERF construction",
                  benchmark_prompts_used=False, legacy_fitted_features_used=False)
    (out / "encoding.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    encode(args.config, args.device)
