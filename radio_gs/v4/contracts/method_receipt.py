"""Frozen v4 stage identity and retained comparator contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RetainedBaselines:
    lerf2d_strict_miou: float = 0.3958417415
    lerf3d_miou: float = 0.437303824780079
    scannet_miou_19: float = 0.3640050732383869
    scannet_miou_15: float = 0.36188714064042804
    scannet_miou_10: float = 0.4671647388007581
    nvos_target_rgb_assisted_iou: float = 0.92555
    lerf2d_target_rgb_assisted_upper_bound: float = 0.576690691074806


@dataclass(frozen=True)
class MethodReceipt:
    stage: str
    carrier: str
    geometry_receipt_sha256: str
    codebook_enabled: bool = False
    query_enabled: bool = False
    compression_enabled: bool = False
    historical_field_opened: bool = False
    target_rgb_opened: bool = False
    schema: str = "radio_gs.surface_object_memory_v4.method_receipt.v1"

    def __post_init__(self) -> None:
        if self.stage == "geometry_registration" and (
            self.codebook_enabled or self.query_enabled or self.compression_enabled
        ):
            raise ValueError("geometry stage cannot enable downstream method modules")
        if self.target_rgb_opened:
            raise ValueError("strict v4 method receipt forbids target RGB")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "retained_baselines": asdict(RetainedBaselines())}
