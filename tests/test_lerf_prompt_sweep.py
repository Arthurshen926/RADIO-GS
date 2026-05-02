from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from radio_gs.scripts.extract_radio_features import (
    _adaptor_output_subdir,
    _compute_scaled_radio_resolution,
    _parse_adaptor_names,
    _stitch_sliding_window_features,
    _unpack_radio_output,
)
from radio_gs.scripts import eval_lerf_grounding as eval_lerf_grounding_module
from radio_gs.scripts.eval_lerf_grounding import load_or_generate_prompt_ensemble_embeddings
from radio_gs.scripts.sweep_lerf_grounding import (
    build_eval_command,
    iter_sweep_cases,
    read_metrics,
    sort_results,
)


def test_sweep_cases_cover_cartesian_product() -> None:
    args = Namespace(
        scoring="softmax_scene,cosine",
        temps="50",
        iou_thresholds="0.4,0.5",
        prompt_templates=["{query}", "a photo of {query}"],
    )

    cases = list(iter_sweep_cases(args))

    assert [(c.scoring, c.temperature, c.iou_threshold) for c in cases] == [
        ("softmax_scene", 50.0, 0.4),
        ("softmax_scene", 50.0, 0.5),
        ("cosine", 50.0, 0.4),
        ("cosine", 50.0, 0.5),
    ]
    assert cases[0].prompt_templates == ("{query}", "a photo of {query}")


def test_build_eval_command_uses_cuda_visible_devices_5() -> None:
    case = next(
        iter_sweep_cases(
            Namespace(
                scoring="softmax_scene",
                temps="35",
                iou_thresholds="0.45",
                prompt_templates=["{query}"],
            )
        )
    )
    args = Namespace(
        config="cfg.yaml",
        checkpoint="ckpt.pth",
        scene="ramen",
        label_dir="/mnt/pool/sqy/3d_understanding/lerf_ovs/label",
        gt_feature_dir=None,
        output_root="out",
        text_embedding_cache="cache.pt",
        heatmap_upsample=4,
        projection_weights="proj.pth",
        summary_head_weights="head.pth",
        python_wrapper="radio_gs/scripts/run_repo_python.sh",
        use_spatial_projection=False,
        save_vis=False,
        gt_only=False,
        gpu=0,
    )

    cmd, env = build_eval_command(args, case, Path("out/run"))

    assert cmd[:4] == ["bash", "radio_gs/scripts/run_repo_python.sh", "-m", "radio_gs.scripts.eval_lerf_grounding"]
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert "--gpu" in cmd
    assert cmd[cmd.index("--gpu") + 1] == "0"
    assert cmd[cmd.index("--label_dir") + 1] == "/mnt/pool/sqy/3d_understanding/lerf_ovs/label"
    assert cmd[cmd.index("--prompt_templates") + 1] == "{query}"


def test_read_metrics_and_sort_prefers_rendered_then_gt(tmp_path: Path) -> None:
    result_path = tmp_path / "lerf_ovs_results.json"
    result_path.write_text(
        json.dumps(
            {
                "scenes": {
                    "ramen": {
                        "gt": {"loc_acc": 0.7, "miou": 0.4, "loc_total": 10},
                        "rendered": {"loc_acc": 0.6, "miou": 0.5, "loc_total": 10},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    metrics = read_metrics(result_path, "ramen")
    rows = sort_results(
        [
            {"rendered_loc_acc": 0.2, "gt_loc_acc": 0.95, "rendered_miou": 0.9, "gt_miou": 0.9},
            metrics,
        ]
    )

    assert metrics["rendered_loc_acc"] == 0.6
    assert metrics["gt_loc_acc"] == 0.7
    assert rows[0] is metrics


def test_resolution_scale_rounds_to_radio_patch_multiple() -> None:
    assert _compute_scaled_radio_resolution(101, 151, 1.5, patch_size=16) == (144, 224)
    assert _compute_scaled_radio_resolution(101, 151, 1.0, patch_size=16) == (96, 144)


def test_stitch_sliding_window_features_blends_overlaps() -> None:
    full = torch.zeros(1, 1, 4, 4)
    tiles = [
        (0, 0, torch.ones(1, 1, 3, 3)),
        (1, 1, torch.full((1, 1, 3, 3), 3.0)),
    ]

    stitched = _stitch_sliding_window_features(full.shape, tiles)

    assert stitched.shape == full.shape
    assert stitched[0, 0, 0, 0].item() == 1.0
    assert stitched[0, 0, 1, 1].item() == 2.0
    assert stitched[0, 0, 3, 3].item() == 3.0


def test_radio_adaptor_names_are_configurable() -> None:
    assert _parse_adaptor_names("siglip2-g,dino_v3,sam3") == [
        "siglip2-g",
        "dino_v3",
        "sam3",
    ]
    assert _adaptor_output_subdir("siglip2-g") == "siglip2"
    assert _adaptor_output_subdir("dino_v3") == "dino_v3"


def test_unpack_radio_output_keeps_requested_adaptors() -> None:
    summary = torch.zeros(1, 2)
    spatial = torch.arange(1 * 4 * 3).reshape(1, 4, 3).float()
    output = (
        summary,
        spatial,
        {
            "siglip2-g": {"spatial": torch.ones(1, 4, 5)},
            "dino_v3": {"spatial": torch.full((1, 4, 7), 2.0)},
        },
    )

    _, _, adaptors = _unpack_radio_output(
        output,
        patch_h=2,
        patch_w=2,
        adaptor_names=["dino_v3"],
    )

    assert list(adaptors) == ["dino_v3"]
    assert adaptors["dino_v3"].shape == (1, 7, 2, 2)
    assert torch.all(adaptors["dino_v3"] == 2.0)


def test_unpack_radio_output_handles_radio_dict_output() -> None:
    output = {
        "backbone": (
            torch.zeros(1, 3),
            torch.arange(1 * 4 * 3).reshape(1, 4, 3).float(),
        ),
        "siglip2-g": (
            torch.zeros(1, 5),
            torch.ones(1, 4, 5),
        ),
        "sam3": (
            torch.zeros(1, 6),
            torch.full((1, 4, 6), 3.0),
        ),
    }

    summary, spatial, adaptors = _unpack_radio_output(
        output,
        patch_h=2,
        patch_w=2,
        adaptor_names=["siglip2-g", "sam3"],
    )

    assert summary.shape == (1, 3)
    assert spatial.shape == (1, 3, 2, 2)
    assert adaptors["siglip2-g"].shape == (1, 5, 2, 2)
    assert adaptors["sam3"].shape == (1, 6, 2, 2)


def test_prompt_ensemble_loads_exact_cache_before_encoding(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "prompt_cache.pt"
    queries = ["wall", "floor"]
    templates = ["{query}", "a photo of a {query}"]
    embeddings = torch.tensor(
        [
            [3.0, 4.0],
            [0.0, 5.0],
        ],
        dtype=torch.float32,
    )
    torch.save(
        {
            "queries": queries,
            "prompt_templates": templates,
            "embeddings": embeddings,
        },
        cache_path,
    )

    encode_calls = []

    def fail_encode(*args, **kwargs):
        encode_calls.append((args, kwargs))
        raise AssertionError("prompt ensemble cache should be used before encoding")

    monkeypatch.setattr(eval_lerf_grounding_module, "encode_text_siglip2", fail_encode)

    loaded = load_or_generate_prompt_ensemble_embeddings(
        queries,
        torch.device("cpu"),
        cache_path=str(cache_path),
        prompt_templates=templates,
    )

    assert torch.allclose(loaded, torch.nn.functional.normalize(embeddings, dim=-1))
    assert encode_calls == []
