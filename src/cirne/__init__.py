"""CIRNE: instance-mask-guided nonrigid registration."""

from .core import (
    RegistrationResult,
    best_translation_by_iou,
    build_coarse_displacement_grid,
    estimate_patch_moves,
    iou_binary,
    mask_metrics,
    register_masks,
    shift2d_zero,
    warp_image,
    warp_labels,
)

__all__ = [
    "RegistrationResult",
    "best_translation_by_iou",
    "build_coarse_displacement_grid",
    "estimate_patch_moves",
    "iou_binary",
    "mask_metrics",
    "register_masks",
    "shift2d_zero",
    "warp_image",
    "warp_labels",
]

__version__ = "0.2.0"
