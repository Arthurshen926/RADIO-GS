from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from radio_gs.scripts.validate_evaluation_protocol_cleanup_inventory import (
    CleanupInventoryError,
    load_and_validate,
    validate_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "paper/artifacts/evaluation_protocol_cleanup_inventory_20260801.yaml"
FREEZE = ROOT / "paper/artifacts/evaluation_protocol_freeze_20260801.yaml"


def test_checked_in_cleanup_inventory_is_non_destructive_and_valid() -> None:
    inventory = load_and_validate(INVENTORY, FREEZE)
    assert inventory["policy"]["deletion_performed"] is False
    assert len(inventory["candidates"]) >= 20


def test_cleanup_candidate_cannot_directly_target_frozen_artifact() -> None:
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    payload = deepcopy(inventory)
    frozen_path = freeze["canonical_tasks"]["spatial_nvos_ludvig"][
        "authoritative_artifacts"
    ][0]["path"]
    payload["candidates"]["bad"] = {
        "category": "safe_remove",
        "target": frozen_path,
        "estimated_size": "small",
        "reason": "invalid test candidate",
    }
    with pytest.raises(CleanupInventoryError, match="frozen artifact"):
        validate_inventory(payload, freeze=freeze)


def test_archive_candidate_requires_retained_receipt_or_blocker() -> None:
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    payload = deepcopy(inventory)
    payload["candidates"]["bad"] = {
        "category": "archive_then_remove",
        "target": "output/historical-run",
        "estimated_size": "1 GB",
        "reason": "invalid test candidate",
    }
    with pytest.raises(CleanupInventoryError, match="retain_before_action or blocker"):
        validate_inventory(payload, freeze=freeze)
