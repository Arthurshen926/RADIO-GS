from pathlib import Path
import os
import subprocess


def test_formal_queue_passes_official_point_eval_args() -> None:
    script = Path("radio_gs/scripts/run_scannet_og_formal_queue.sh").read_text(
        encoding="utf-8"
    )

    assert 'EVAL_QUERY_MODE="${EVAL_QUERY_MODE:-${DIRECT_POINT_QUERY_MODE}}"' in script
    assert 'EVAL_OPACITY_FILTER_MODE="${EVAL_OPACITY_FILTER_MODE:-label_index}"' in script
    assert '--query_mode "${EVAL_QUERY_MODE}"' in script
    assert '--opacity_filter_mode "${EVAL_OPACITY_FILTER_MODE}"' in script


def test_v57_v59_chain_script_has_dry_run_and_no_background_waiter() -> None:
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    result = subprocess.run(
        [
            "bash",
            "radio_gs/scripts/run_scannet_og_v57_v59_chain.sh",
            "4",
            "scene0000_00",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "DRY_RUN=1" in result.stdout
    assert "START scene0000_v57_train" in result.stdout
    assert "START scene0000_v57_eval" in result.stdout
    assert "START scene0000_teacher_cache_norm" in result.stdout
    assert "START scene0000_v59_train" in result.stdout
    assert "START scene0000_v59_eval" in result.stdout
    assert (
        "output/scannet_teacher_cache_norm/\\{scene\\}_radio_teacher_features.pt"
        in result.stdout
    )
    assert "\\{query\\}\\|a\\ photo\\ of\\ a" in result.stdout

    script = Path("radio_gs/scripts/run_scannet_og_v57_v59_chain.sh").read_text(
        encoding="utf-8"
    )
    assert "wait_and_run.sh" not in script
    assert "nohup" not in script
    assert "trap cleanup EXIT" in script
