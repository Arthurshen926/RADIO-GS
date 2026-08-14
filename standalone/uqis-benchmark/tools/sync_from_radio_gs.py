#!/usr/bin/env python3
"""Refresh the standalone UQIS source snapshot from RADIO-GS canonical files."""

from __future__ import annotations

from pathlib import Path
import shutil


PACKAGE_FILES = (
    "protocol.py",
    "metrics.py",
    "construction.py",
    "scannet_assets.py",
    "official_constructor.py",
    "construction_authority.py",
    "method_fields.py",
    "evaluate_predictions.py",
    "controlled_evaluation.py",
    "workspace.py",
    "seal_predictions.py",
    "audit_benchmark.py",
    "audit_text_annotations.py",
    "build_benchmark.py",
    "stage_query_workspace.py",
)
TEST_FILES = (
    "test_scannet_uqis_protocol.py",
    "test_scannet_uqis_evaluation.py",
    "test_scannet_uqis_method_fields.py",
    "test_scannet_uqis_construction_authority.py",
)
ADR_FILES = (
    "0001-evaluator-private-unified-query-pairing.md",
    "0002-account-modality-specific-multiple-fields.md",
    "0003-separate-construction-and-evaluation-authority.md",
    "0004-separate-core-grounding-from-relational-text.md",
    "0005-inventory-modality-field-dependency-sets.md",
)


def main() -> None:
    standalone = Path(__file__).resolve().parents[1]
    repository = standalone.parents[1]
    canonical = repository / "radio_gs/benchmarks/scannet_uqis"
    package = standalone / "src/uqis_benchmark"
    tests = standalone / "tests"
    adr_output = standalone / "docs/adr"
    ludvig_example = standalone / "examples/ludvig"
    package.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    adr_output.mkdir(parents=True, exist_ok=True)
    ludvig_example.mkdir(parents=True, exist_ok=True)
    for name in PACKAGE_FILES:
        source = canonical / name
        text = source.read_text(encoding="utf-8")
        if "radio_gs." in text:
            raise ValueError(f"standalone source still imports RADIO-GS: {name}")
        (package / name).write_text(text, encoding="utf-8")
    for name in TEST_FILES:
        text = (repository / "tests" / name).read_text(encoding="utf-8")
        text = text.replace("radio_gs.benchmarks.scannet_uqis", "uqis_benchmark")
        (tests / name.replace("test_scannet_uqis_", "test_")).write_text(
            text, encoding="utf-8"
        )
    for name in ADR_FILES:
        shutil.copyfile(repository / "docs/adr" / name, adr_output / name)
    shutil.copyfile(
        canonical / "ludvig_text_diffusion.py",
        ludvig_example / "text_diffusion.py",
    )
    print(
        f"synced {len(PACKAGE_FILES)} modules, {len(TEST_FILES)} tests, "
        f"{len(ADR_FILES)} ADRs, and the pure LUDVIG diffusion example"
    )


if __name__ == "__main__":
    main()
