import importlib
from pathlib import Path

import pytest
import yaml


def _load_validator():
    try:
        return importlib.import_module("radio_gs.scripts.validate_paper_claims")
    except ImportError as exc:
        pytest.fail(f"missing validate_paper_claims module: {exc}")


def _write_good_fixture(root: Path) -> None:
    (root / "paper/artifacts").mkdir(parents=True)
    (root / "docs").mkdir()

    (root / "paper/lerf_direct_3d_context_table.tex").write_text(
        "\\method{} + VPR & local, SigLIP2, fixed \\texttt{thr0p25} + RGB snap & 48.01 & 67.60 \\\\\n"
        "\\method{} + direct field + SAM3 box & local diagnostic, scene-locked thresholds & 59.72 & 70.09 \\\\\n",
        encoding="utf-8",
    )
    (root / "paper/vpr_protocol_card.tex").write_text(
        "Selection & Fixed global softmax-score threshold \\texttt{thr0p25} "
        "with 0.5\\% floor and 1.8\\% cap; mean+std and fixed-ratio sweeps "
        "are diagnostic only \\\\\n",
        encoding="utf-8",
    )
    (root / "paper/lerf_direct_3d_selection_table.tex").write_text(
        "CTF-GS & thr0p25 mIoU & 0.531 & 0.580 & 0.566 & 0.243 & 0.480 \\\\\n"
        "CTF-GS & thr0p25 Acc@0.25 & 0.786 & 0.746 & 0.763 & 0.409 & 0.676 \\\\\n"
        "CTF-GS & SAM3 box fixed thr0p25 Acc@0.25 & 0.696 & 0.746 & 0.746 & 0.546 & 0.684 \\\\\n",
        encoding="utf-8",
    )
    (root / "paper/scannet_published_context_table.tex").write_text(
        "LangSplatV2 & 14.75 & 25.47 & 17.09 & 35.68 & 22.83 & 41.52 \\\\\n"
        "VALA & 32.11 & 50.05 & 35.10 & 54.77 & 46.21 & 65.61 \\\\\n"
        "\\method{} DINO-CV contextual kNN + spatial propagation & \\textbf{38.06} & \\textbf{61.29} & \\textbf{38.71} & \\textbf{63.15} & \\textbf{47.11} & \\textbf{72.00} \\\\\n",
        encoding="utf-8",
    )
    (root / "paper/artifacts/final_rows.yaml").write_text(
        yaml.safe_dump(
            {
                "tracks": {
                    "t2_lerf_direct_3d_selection": {
                        "protocol": {
                            "main_selector_policy": (
                                "fixed_global_softmax_threshold:thr0p25"
                            )
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "paper/radio_gs_draft.tex").write_text(
        "This is not presented as a full ScanNet semantic segmentation leaderboard result.\n"
        "We use the result as evidence rather than as a universal primitive-level SOTA claim.\n",
        encoding="utf-8",
    )
    (root / "docs/paper_draft_current.md").write_text(
        "This should be framed as feature usability evidence, not as a fully standard "
        "ScanNet semantic segmentation leaderboard comparison.\n"
        "The paper should still avoid implying raw Gaussian-center\n"
        "superiority or global direct-3D SOTA.\n"
        "The direct compact field has a confidence-weighted transfer row.\n",
        encoding="utf-8",
    )
    (root / "docs/submission_status.md").write_text(
        "The context table prevents overclaiming global direct-3D SOTA.\n",
        encoding="utf-8",
    )


def test_validate_claims_accepts_fixed_threshold_and_negated_scannet_claims(tmp_path):
    validator = _load_validator()
    _write_good_fixture(tmp_path)

    issues = validator.validate_claims(root=tmp_path)

    assert issues == []


def test_validate_claims_flags_mean_plus_2p5std_vpr_promotion(tmp_path):
    validator = _load_validator()
    _write_good_fixture(tmp_path)
    (tmp_path / "paper/lerf_direct_3d_context_table.tex").write_text(
        "\\method{} + VPR & local, SigLIP2, mean+2.5std + RGB snap & 48.01 & 67.60 \\\\\n",
        encoding="utf-8",
    )

    issues = validator.validate_claims(root=tmp_path)

    assert any("VPR context row" in issue and "thr0p25" in issue for issue in issues)


def test_validate_claims_flags_positive_scannet_leaderboard_claim(tmp_path):
    validator = _load_validator()
    _write_good_fixture(tmp_path)
    (tmp_path / "paper/radio_gs_draft.tex").write_text(
        "This is presented as a full ScanNet semantic segmentation leaderboard result.\n",
        encoding="utf-8",
    )

    issues = validator.validate_claims(root=tmp_path)

    assert any("full ScanNet semantic segmentation leaderboard" in issue for issue in issues)


def test_validate_claims_flags_cags_in_opengaff_context_table(tmp_path):
    validator = _load_validator()
    _write_good_fixture(tmp_path)
    (tmp_path / "paper/lerf_direct_3d_context_table.tex").write_text(
        "CAGS~\\cite{sun2025cags} & published context & 50.79 & 69.62 \\\\\n"
        "\\method{} + VPR & local, SigLIP2, fixed \\texttt{thr0p25} + RGB snap & 48.01 & 67.60 \\\\\n",
        encoding="utf-8",
    )

    issues = validator.validate_claims(root=tmp_path)

    assert any("CAGS must not be promoted" in issue for issue in issues)


def test_validate_claims_flags_cags_in_scannet_context_table(tmp_path):
    validator = _load_validator()
    _write_good_fixture(tmp_path)
    (tmp_path / "paper/scannet_published_context_table.tex").write_text(
        "CAGS & 35.00 & 50.00 & 37.00 & 55.00 & 47.00 & 66.00 \\\\\n"
        "LangSplatV2 & 14.75 & 25.47 & 17.09 & 35.68 & 22.83 & 41.52 \\\\\n"
        "VALA & 32.11 & 50.05 & 35.10 & 54.77 & 46.21 & 65.61 \\\\\n"
        "\\method{} DINO-CV contextual kNN + spatial propagation & \\textbf{38.06} & \\textbf{61.29} & \\textbf{38.71} & \\textbf{63.15} & \\textbf{47.11} & \\textbf{72.00} \\\\\n",
        encoding="utf-8",
    )

    issues = validator.validate_claims(root=tmp_path)

    assert any("ScanNet table" in issue and "CAGS" in issue for issue in issues)


def test_validate_claims_flags_exact_opengaff_scannet_reproduction_claim(tmp_path):
    validator = _load_validator()
    _write_good_fixture(tmp_path)
    (tmp_path / "paper/radio_gs_draft.tex").write_text(
        "This is an exact OpenGaFF ScanNet reproduction.\n",
        encoding="utf-8",
    )

    issues = validator.validate_claims(root=tmp_path)

    assert any("exact OpenGaFF ScanNet reproduction" in issue for issue in issues)
