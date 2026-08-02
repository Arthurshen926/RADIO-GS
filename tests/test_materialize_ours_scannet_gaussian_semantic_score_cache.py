from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from radio_gs.scripts.eval_ours_scannet_vala_gaussian_protocol import (
    PAPER_CLASS_IDS,
    PAPER_CLASS_NAMES,
    load_ours_gaussian_semantic_score_cache,
)
from radio_gs.scripts import (
    materialize_ours_scannet_gaussian_semantic_score_cache as materializer,
)
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


class _FakeModel:
    def __init__(self) -> None:
        self.xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
        )
        self.scale = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.2, 0.2], [0.3, 0.4, 0.5]],
            dtype=torch.float32,
        )
        self.rotation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float32
        )
        self.opacity = torch.tensor([[0.5], [0.75], [1.0]], dtype=torch.float32)
        self.queries: list[tuple[list[int], object]] = []

    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    def get_scaling(self) -> torch.Tensor:
        return self.scale

    def get_rotation(self) -> torch.Tensor:
        return self.rotation

    def get_opacity(self) -> torch.Tensor:
        return self.opacity

    def query_gaussian_points(
        self,
        gaussian_indices: torch.Tensor,
        *,
        points_xyz=None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        assert return_aux is False
        rows = gaussian_indices.detach().cpu().tolist()
        self.queries.append((rows, points_xyz))
        compact = torch.zeros(len(rows), 2, device=gaussian_indices.device)
        compact[:, 0] = torch.as_tensor(rows, device=gaussian_indices.device) + 1.0
        compact[:, 1] = 1.0
        return compact


class _FakeCodec:
    def decode_points(self, compact: torch.Tensor) -> torch.Tensor:
        decoded = torch.zeros(compact.shape[0], 1280, device=compact.device)
        decoded[:, :2] = compact
        return decoded


class _FakeSummaryHead(torch.nn.Module):
    def forward(self, decoded: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            decoded.shape[0], decoded.shape[1], 1536, device=decoded.device
        )
        result[..., :2] = decoded[..., :2]
        return result


class _FakeCanonicalField(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_gaussians = 3
        self.calls: list[list[int]] = []

    def radio_features(self, rows: torch.Tensor) -> torch.Tensor:
        indices = rows.detach().cpu().tolist()
        self.calls.append(indices)
        values = torch.zeros(len(indices), 1280, device=rows.device)
        for offset, row in enumerate(indices):
            values[offset, 0] = float(row + 1)
            values[offset, 1] = float(3 - row)
        return values


class _FakeRegionReadout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        radio: torch.Tensor,
        _geometry: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        weights = token_mask.float() * reliability[..., 0]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bt,btc->bc", weights, radio.float())


def _write_query_source(path: Path, *, queries=None) -> None:
    embeddings = torch.zeros(len(PAPER_CLASS_IDS), 1536, dtype=torch.float32)
    embeddings[:, 0] = torch.linspace(1.0, 2.0, len(PAPER_CLASS_IDS))
    embeddings[:, 1] = torch.linspace(2.0, 1.0, len(PAPER_CLASS_IDS))
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    torch.save(
        {
            "queries": list(PAPER_CLASS_NAMES) if queries is None else queries,
            "embeddings": embeddings,
            "model_name": materializer.SIGLIP2_MODEL_NAME,
            "text_encoder": "siglip2",
            "exact_scannet_nyu40": True,
            "head_fixed": True,
            "prompt_templates": ["{query}"],
        },
        path,
    )


def _args(tmp_path: Path) -> Namespace:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "best.pth"
    query = tmp_path / "split19.pt"
    summary = tmp_path / "summary_head.pth"
    config.write_text("scene: scene0000_00\n", encoding="utf-8")
    checkpoint.write_bytes(b"trained geometry, field, and codec authority")
    summary.write_bytes(b"official extracted summary head authority")
    _write_query_source(query)
    return Namespace(
        scene="scene0000_00",
        method_family=materializer.LEGACY_METHOD_FAMILY,
        config=str(config),
        geometry_checkpoint=str(checkpoint),
        query_source=str(query),
        summary_head_weights=str(summary),
        output=str(tmp_path / "scores.pt"),
        receipt=str(tmp_path / "receipt.json"),
        device="cpu",
        chunk_size=2,
    )


def test_materialize_builds_evaluator_valid_row_bound_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    model = _FakeModel()
    monkeypatch.setattr(
        materializer, "load_config", lambda _path: SimpleNamespace(scene=args.scene)
    )
    monkeypatch.setattr(
        materializer, "_build_hybrid_model", lambda _cfg, _path, _device: (model, _FakeCodec())
    )
    monkeypatch.setattr(
        materializer.SigLIP2SummaryHead,
        "from_extracted_weights",
        lambda _path: _FakeSummaryHead(),
    )

    output, receipt = materializer.materialize(args)
    cache, digest, source = load_ours_gaussian_semantic_score_cache(
        output,
        expected_scene_id=args.scene,
        expected_xyz=model.xyz,
        expected_scale=model.scale,
        expected_quaternion=model.rotation,
        expected_opacity=model.opacity.reshape(-1),
        expected_valid=torch.ones(3, dtype=torch.bool),
        expected_geometry_checkpoint_sha256=sha256_file(args.geometry_checkpoint),
        expected_method_family=materializer.LEGACY_METHOD_FAMILY,
    )
    receipt_payload, _, _ = load_json_object(receipt)

    assert source == output.resolve()
    assert digest == sha256_file(output)
    assert cache["semantic_scores"].shape == (3, 19)
    assert cache["metadata"]["semantic_source_sha256"] == sha256_file(
        args.geometry_checkpoint
    )
    assert cache["metadata"]["summary_head_source"]["sha256"] == sha256_file(
        args.summary_head_weights
    )
    assert cache["metadata"]["gaussian_query_position"] == "optimized_gaussian_center"
    assert cache["metadata"]["logit_calibration"] == "none"
    assert cache["metadata"]["knn_used"] is False
    assert cache["metadata"]["protocol_freeze_id"] == materializer.PROTOCOL_FREEZE_ID
    assert (
        cache["metadata"]["protocol_freeze_sha256"]
        == materializer.PROTOCOL_FREEZE_SHA256
    )
    assert cache["metadata"]["producer_source_sha256"] == sha256_file(
        materializer.__file__
    )
    assert receipt_payload["semantic_score_cache"]["sha256"] == digest
    assert model.queries == [([0, 1], None), ([2], None)]


def test_text_bank_rejects_wrong_query_order_and_zero_rows(tmp_path: Path) -> None:
    path = tmp_path / "wrong.pt"
    _write_query_source(path, queries=list(reversed(PAPER_CLASS_NAMES)))
    with pytest.raises(ValueError, match="query/order differs"):
        materializer.load_frozen_split19_text_bank(path, device=torch.device("cpu"))

    path = tmp_path / "zero.pt"
    _write_query_source(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["embeddings"][3].zero_()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="zero-norm"):
        materializer.load_frozen_split19_text_bank(path, device=torch.device("cpu"))


def test_compute_scores_rejects_invalid_chunk_and_uses_every_row() -> None:
    model = _FakeModel()
    text = torch.zeros(19, 1536)
    text[:, 0] = 1.0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        materializer.compute_gaussian_semantic_scores(
            model,
            _FakeCodec(),
            _FakeSummaryHead(),
            text,
            device=torch.device("cpu"),
            chunk_size=0,
        )
    scores = materializer.compute_gaussian_semantic_scores(
        model,
        _FakeCodec(),
        _FakeSummaryHead(),
        text,
        device=torch.device("cpu"),
        chunk_size=1,
    )
    assert scores.shape == (3, 19)
    assert model.queries == [([0], None), ([1], None), ([2], None)]


def test_canonical_geometry_authority_is_cpu_deterministic() -> None:
    scoring_device = torch.device("cuda:0")
    assert materializer._geometry_authority_device(
        materializer.CURRENT_METHOD_FAMILY, scoring_device
    ) == torch.device("cpu")
    assert materializer._geometry_authority_device(
        materializer.LEGACY_METHOD_FAMILY, scoring_device
    ) == scoring_device


def test_canonical_totality_routes_observed_regions_and_no_evidence_fallback() -> None:
    field = _FakeCanonicalField()
    readout = _FakeRegionReadout()
    mpr = {
        "reliability": torch.ones(3, 2),
    }
    graph = {
        "global_rows": torch.tensor([0, 2]),
        "xyz": torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "raw_affinity": torch.ones(2),
        "local_sigma": torch.full((2,), 0.05),
    }
    text = torch.zeros(19, 1536)
    text[:, 0] = torch.linspace(1.0, 2.0, 19)
    text[:, 1] = torch.linspace(2.0, 1.0, 19)
    text = torch.nn.functional.normalize(text, dim=-1)

    scores, observed = materializer.compute_canonical_mpr_v3_semantic_scores(
        field,
        mpr,
        graph,
        readout,
        _FakeSummaryHead(),
        text,
        device=torch.device("cpu"),
        radio_batch_size=8,
        semantic_batch_size=2,
    )

    direct_row_one = torch.nn.functional.normalize(
        torch.tensor([[2.0, 2.0]]), dim=-1
    ) @ text[:, :2].T
    assert observed.tolist() == [True, False, True]
    assert scores.shape == (3, 19) and torch.isfinite(scores).all()
    torch.testing.assert_close(scores[1], direct_row_one[0], atol=1e-6, rtol=1e-6)
    assert field.calls == [[0, 2], [1]]
    assert readout.calls == len(materializer.CANONICAL_REGION_RADII_M)


def test_materialize_refuses_to_overwrite_before_loading_model(tmp_path: Path) -> None:
    args = _args(tmp_path)
    Path(args.output).write_bytes(b"existing immutable result")
    with pytest.raises(FileExistsError, match="immutable output already exists"):
        materializer.materialize(args)
