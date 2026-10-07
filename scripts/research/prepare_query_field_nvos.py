"""Prepare the predeclared fern development probe without opening target labels."""
import json
from pathlib import Path

from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.query_field.storage import sha256
import torch


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def main():
    root = Path("output/query_field_20261007/nvos_fern_v1")
    root.mkdir(parents=True, exist_ok=False)
    config_root = Path("configs/query_field")
    encoder = json.loads(Path("output/system_map_rebuild/figurines/complete/scene.json").read_text())["text_encoder"]
    audit = []
    for scene_path in sorted(Path("configs/pipeline/nvos_all_sources").glob("*.scene.json")):
        scene = json.loads(scene_path.read_text())
        key = scene["scene"]
        query = json.loads(Path(f"configs/consolidated/nvos/{key}.query.json").read_text())
        positive = Path(query["positive_scribble"])
        negative = positive.with_name(positive.name.replace("pos_", "neg_", 1))
        cache = Path("output/nvos_source_sam_20260917") / key / "manifest.json"
        images = json.loads(cache.read_text())["images"] if cache.exists() else []
        by_rgb = {row["image"]: row for row in images}
        frames = [f for f in scene["source_frames"] if scene["source_rgb"][str(f)] in by_rgb]
        audit.append(dict(scene=key, source_frames=len(scene["source_frames"]),
                          verified_provider_candidate_frames=frames,
                          positive_exists=positive.is_file(), negative_exists=negative.is_file(),
                          manifest=None if not cache.exists() else str(cache),
                          scope="designated query-free provider only; not exhaustive legacy filesystem search"))
        if key != "fern":
            continue
        if len(frames) != 14:
            raise ValueError("predeclared cache availability changed")
        development = [frames[len(frames)//4], frames[3*len(frames)//4]]
        train = [f for f in frames if f not in development]
        entries = [dict(frame=f, rgb=scene["source_rgb"][str(f)],
                        rgb_sha256=by_rgb[scene["source_rgb"][str(f)]]["source_image_sha256"],
                        sam=by_rgb[scene["source_rgb"][str(f)]]["output"],
                        sam_sha256=by_rgb[scene["source_rgb"][str(f)]]["output_sha256"]) for f in frames]
        regions = root / "native_regions"
        write(config_root / "nvos_fern_regions.json", dict(output=str(regions), encoder=encoder, frames=entries))
        authority = root / "source_authority.json"
        write(authority, dict(images=[dict(image_id=f"frame_{r['frame']:05d}", path=r["rgb"]) for r in entries]))
        clean_scene = {k: scene[k] for k in ["scene", "geometry", "camera_report", "camera_report_sha256"]}
        clean_scene.update(source_frames=frames, source_rgb_authority=str(authority), text_encoder=encoder,
                           native_dimension=1536, native_region_directories=[str(regions)])
        write(root / "scene.json", clean_scene)
        config = json.loads((config_root / "figurines_spatial_metric.json").read_text())
        config.update(scene=str(root / "scene.json"), train_frames=train, development_frames=development,
                      purpose="First real signed NVOS development probe, unchanged common physical member field",
                      selection="All 14 cached authorized query-free frames; sorted roster quartiles held out; availability selection before scoring",
                      input_budget_condition="12 native training views, 2 teacher development views; existing frozen geometry")
        write(config_root / "nvos_fern_physical.json", config)
        evaluation = json.loads(Path("configs/consolidated/nvos/fern.evaluation.json").read_text())
        if evaluation["target_frame"] in scene["source_frames"]:
            raise ValueError("target frame in construction authority")
        cameras = MogeGeometryPrior(scene["camera_report"], scene["camera_report_sha256"])
        fixture = {}
        for role, frame in [("reference", query["reference_frame"]), ("target", evaluation["target_frame"])]:
            c = cameras.cameras[frame]
            fixture[role] = dict(intrinsic=c.intrinsic, world_to_camera=c.world_to_camera,
                                 height=c.height, width=c.width, frame=frame)
            if role == "reference":
                width, height = query["prompt_raster_wh"]
                intrinsic = c.intrinsic.clone()
                intrinsic[0] *= width/c.width
                intrinsic[1] *= height/c.height
                fixture[role].update(intrinsic=intrinsic, height=height, width=width)
        fixture["world_units_per_metre"] = cameras.scale
        torch.save(fixture, root / "cameras.pt")
        jobs = [dict(name=f"{stage}_{mode}_{scope}", field=str(root / "models" / stage), mode=mode, scope=scope,
                     positive=str(positive), negative=str(negative), camera_fixture=str(root / "cameras.pt"))
                for stage in ["untrained", "trained"] for mode in ["positive", "signed"]
                for scope in ["default", "positive_extent"]]
        prediction = dict(schema="radio_gs.query_field_nvos.v1", scene="fern", jobs=jobs,
                          projection_hw=[192, 256], output_hw=[756, 1008], alpha_threshold=.5,
                          prompt_resize="area evidence pooling; no pixel label threshold",
                          prediction_policy="common point member decoder; default scale vs max(default, 2*positive ray-centre spatial std)",
                          evaluation_status="historically opened development, 1 of 8 cases",
                          comparison="v10 positive-only source-assisted SAM-derived reference uses a different contract")
        prediction["readouts"] = dict(actual_subset="192x256 selected alpha -> bilinear 756x1008 -> >0.5",
                                      retained_response="full-scene observed-support-normalized response at 756x1008 -> >=0.5 and AP")
        write(config_root / "nvos_fern_prediction.json", prediction)
        write(config_root / "nvos_fern_scoring.json", dict(scene="fern", gt_mask=evaluation["gt_mask"],
                                                        prediction_config=str(config_root / "nvos_fern_prediction.json")))
    write(root / "availability.json", dict(cases=audit, target_labels_opened=False))
    print(json.dumps(dict(output=str(root), train_frames=train, development_frames=development), indent=2))


if __name__ == "__main__":
    main()
