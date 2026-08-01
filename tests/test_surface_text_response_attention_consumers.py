from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from radio_gs.scripts import finalize_surface_text_response_promotion as promotion
from radio_gs.scripts import materialize_surface_text_response_descriptors as materializer
from radio_gs.scripts import surface_text_response_distill_authority as authority


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMOTION_RUNNER = REPO_ROOT / "radio_gs/scripts/run_surface_text_response_promotion.sh"
CURRENT_POSTCACHE_ROOT = Path(
    "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/"
    "surface_c1024_attention_postcache_v1_gpu1only_src1b85cfdaf7b5"
)
JOINT = "joint_attention_v1"
CANDIDATE = "context_c1024_geometric"


def _load_attention_fixture_module() -> ModuleType:
    path = REPO_ROOT / "tests/test_surface_text_response_distill_attention_binding.py"
    spec = importlib.util.spec_from_file_location("attention_binding_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATTENTION_FIXTURE = _load_attention_fixture_module()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _selected_checkpoint(fixture: dict[str, object], seed: int = 0) -> dict:
    screen = _load_json(fixture["screen_path"])
    matches = [
        row
        for row in screen["variants"][JOINT]["seeds"]
        if row["seed"] == seed
    ]
    assert len(matches) == 1
    return matches[0]["checkpoint"]


def _materializer_binding(
    fixture: dict[str, object],
    *,
    seed: int = 0,
    cache_bindings: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    checkpoint = _selected_checkpoint(fixture, seed)
    checkpoint_path = Path(checkpoint["path"])
    return materializer._validate_attention_postcache_binding(
        Path(fixture["screen_path"]),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint["sha256"],
        report_path=checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"),
        seed=seed,
        cache_meta={
            "cache_bindings": (
                list(fixture["validation"])
                if cache_bindings is None
                else cache_bindings
            )
        },
    )


def test_materializer_accepts_synthetic_attention_postcache_binding(
    tmp_path: Path,
) -> None:
    fixture = ATTENTION_FIXTURE._postcache_fixture(tmp_path)

    binding = _materializer_binding(fixture)

    assert binding == {
        "path": str(Path(fixture["screen_path"]).resolve()),
        "sha256": authority.sha256_file(fixture["screen_path"]),
        "completion": str(Path(fixture["completion_path"]).resolve()),
        "completion_sha256": authority.sha256_file(fixture["completion_path"]),
        "candidate": CANDIDATE,
    }


@pytest.mark.skipif(
    not (CURRENT_POSTCACHE_ROOT / "screen.complete").is_file(),
    reason="formal post-cache continuation is not mounted",
)
def test_materializer_accepts_current_formal_attention_postcache() -> None:
    screen_path = CURRENT_POSTCACHE_ROOT / "attention_pooling_screen.json"
    screen = _load_json(screen_path)
    selected = screen["variants"][JOINT]["seeds"][0]["checkpoint"]
    checkpoint_path = Path(selected["path"])
    pairing = _load_json(CURRENT_POSTCACHE_ROOT / "cache_pairing.json")
    validation = [
        row["c1024"]
        for row in pairing["rows"]
        if row["role"] == "validation"
    ]

    binding = materializer._validate_attention_postcache_binding(
        screen_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=selected["sha256"],
        report_path=checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"),
        seed=0,
        cache_meta={"cache_bindings": validation},
    )

    assert binding["candidate"] == CANDIDATE
    assert binding["path"] == str(screen_path.resolve())


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        ("screen_path", "benchmark_queries_opened"),
        ("screen_path", "benchmark_masks_opened"),
        ("pairing_path", "benchmark_queries_opened"),
        ("pairing_path", "benchmark_masks_opened"),
    ],
)
def test_materializer_rejects_open_attention_query_or_mask_evidence(
    tmp_path: Path,
    artifact: str,
    field: str,
) -> None:
    fixture = ATTENTION_FIXTURE._postcache_fixture(tmp_path)
    path = Path(fixture[artifact])
    payload = _load_json(path)
    payload[field] = True
    _write_json(path, payload)

    with pytest.raises(ValueError):
        _materializer_binding(fixture)


def test_materializer_rejects_missing_joint_seed(tmp_path: Path) -> None:
    fixture = ATTENTION_FIXTURE._postcache_fixture(tmp_path)
    screen_path = Path(fixture["screen_path"])
    screen = _load_json(screen_path)
    screen["variants"][JOINT]["seeds"].pop()
    _write_json(screen_path, screen)

    with pytest.raises(ValueError):
        _materializer_binding(fixture)


def test_materializer_rejects_missing_c1024_cache_row(tmp_path: Path) -> None:
    fixture = ATTENTION_FIXTURE._postcache_fixture(tmp_path)
    pairing_path = Path(fixture["pairing_path"])
    pairing = _load_json(pairing_path)
    pairing["rows"].pop()
    _write_json(pairing_path, pairing)

    with pytest.raises(ValueError):
        _materializer_binding(fixture)


def test_materializer_rejects_validation_cache_consumer_mismatch(
    tmp_path: Path,
) -> None:
    fixture = ATTENTION_FIXTURE._postcache_fixture(tmp_path)
    mismatched = list(fixture["validation"])[1:]

    with pytest.raises(ValueError):
        _materializer_binding(fixture, cache_bindings=mismatched)


@pytest.mark.parametrize("artifact", ["checkpoint", "cache"])
def test_materializer_rejects_attention_artifact_sha_drift(
    tmp_path: Path,
    artifact: str,
) -> None:
    fixture = ATTENTION_FIXTURE._postcache_fixture(tmp_path)
    if artifact == "checkpoint":
        path = Path(_selected_checkpoint(fixture)["path"])
    else:
        path = Path(fixture["validation"][0]["path"])
    path.write_bytes(path.read_bytes() + b"-sha-drift")

    with pytest.raises(ValueError):
        _materializer_binding(fixture)


@pytest.mark.skipif(
    not (CURRENT_POSTCACHE_ROOT / "screen.complete").is_file(),
    reason="formal post-cache continuation is not mounted",
)
def test_promotion_accepts_current_attention_screen_with_cpu_recomputation() -> None:
    screen_path = CURRENT_POSTCACHE_ROOT / "attention_pooling_screen.json"
    completion_path = CURRENT_POSTCACHE_ROOT / "screen.complete"
    pairing = _load_json(CURRENT_POSTCACHE_ROOT / "cache_pairing.json")
    rows = pairing["rows"]
    train = [row["c1024"] for row in rows if row["role"] == "train"]
    validation = [row["c1024"] for row in rows if row["role"] == "validation"]

    surface = promotion._validate_surface_bundle(screen_path, completion_path)
    adapter = authority._surface_binding(
        surface_root=CURRENT_POSTCACHE_ROOT,
        candidate=CANDIDATE,
        train=train,
        validation=validation,
    )

    assert surface["selected_candidate"] == CANDIDATE
    assert surface["distill_surface_promotion"] == adapter
    assert set(surface["selected_by_seed"]) == {0, 1, 2}
    assert all(
        row["candidate"] == CANDIDATE
        for row in surface["selected_by_seed"].values()
    )
    assert len(surface["selected_caches"]["train"]) == 4
    assert len(surface["selected_caches"]["validation"]) == 2


def test_promotion_runner_maps_attention_winner_and_is_cpu_only() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(PROMOTION_RUNNER)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    source = PROMOTION_RUNNER.read_text(encoding="utf-8")

    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert "nvidia-smi" not in source
    assert "surface_c1024_attention_pooling_postcache_continuation" in source
    assert 'payload.get("selected_variant") == "joint_attention_v1"' in source
    assert 'payload.get("selection_status") == "joint_attention_retained"' in source
    assert 'selected = "context_c1024_geometric"' in source
    assert '--readout-binding-manifest "$PROMOTION_MANIFEST"' in source
    assert "--device cpu" in source

