"""Numbers only: no historical method implementation is imported here."""

RETAINED_BASELINES = {
    "lerf2d_strict_miou": 0.3958417415,
    "lerf3d_miou": 0.437303824780079,
    "scannet_miou_19_15_10": (0.3640050732383869, 0.36188714064042804, 0.4671647388007581),
    "nvos_rgb_assisted_macro_iou": 0.92555,
    "lerf2d_target_rgb_assisted_upper_bound": 0.576690691074806,
}

__all__ = ["RETAINED_BASELINES"]
