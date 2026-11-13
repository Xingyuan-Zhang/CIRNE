"""Core CIRNE registration algorithm.

CIRNE estimates local integer translations from binary foreground occupancy,
interpolates those translations into a smooth displacement field, and applies
the field to the original instance labels with nearest-neighbour sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class RegistrationResult:
    """Sparse translations and their interpolated coarse displacement field."""

    patch_moves: dict[str, np.ndarray]
    dy_grid: np.ndarray
    dx_grid: np.ndarray
    grid_step: int
    shape: tuple[int, int]


def shift2d_zero(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2-D array, filling exposed pixels with zero."""
    if array.ndim != 2:
        raise ValueError("shift2d_zero expects a 2-D array")
    height, width = array.shape
    out = np.zeros_like(array)
    source_y0, source_y1 = max(0, -dy), min(height, height - dy)
    source_x0, source_x1 = max(0, -dx), min(width, width - dx)
    target_y0, target_y1 = max(0, dy), min(height, height + dy)
    target_x0, target_x1 = max(0, dx), min(width, width + dx)
    if source_y1 > source_y0 and source_x1 > source_x0:
        out[target_y0:target_y1, target_x0:target_x1] = array[
            source_y0:source_y1, source_x0:source_x1
        ]
    return out


def iou_binary(first: np.ndarray, second: np.ndarray) -> float:
    """Return intersection over union for two masks."""
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} and {second.shape}")
    intersection = np.logical_and(first, second).sum(dtype=np.int64)
    union = np.logical_or(first, second).sum(dtype=np.int64)
    return 1.0 if union == 0 else float(intersection) / float(union)


def mask_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """Compute common binary overlap metrics."""
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} and {second.shape}")
    tp = int(np.logical_and(first, second).sum(dtype=np.int64))
    fp = int(np.logical_and(first, ~second).sum(dtype=np.int64))
    fn = int(np.logical_and(~first, second).sum(dtype=np.int64))
    tn = int(np.logical_and(~first, ~second).sum(dtype=np.int64))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_translation_by_iou(
    moving: np.ndarray,
    fixed: np.ndarray,
    refine_radius: int = 6,
) -> Optional[tuple[int, int, float]]:
    """Estimate the integer ``(dy, dx)`` that maps moving onto fixed."""
    moving = (np.asarray(moving) > 0).astype(np.uint8)
    fixed = (np.asarray(fixed) > 0).astype(np.uint8)
    if moving.shape != fixed.shape:
        raise ValueError(f"Shape mismatch: {moving.shape} and {fixed.shape}")
    if refine_radius < 0:
        raise ValueError("refine_radius must be non-negative")
    if not moving.any() or not fixed.any():
        return None

    correlation = fftconvolve(
        moving.astype(np.float32), fixed[::-1, ::-1].astype(np.float32), mode="full"
    )
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    lag_y = int(peak_y - (fixed.shape[0] - 1))
    lag_x = int(peak_x - (fixed.shape[1] - 1))

    best_dy, best_dx, best_score = 0, 0, -1.0
    for center_y, center_x in {(lag_y, lag_x), (-lag_y, -lag_x)}:
        for dy in range(center_y - refine_radius, center_y + refine_radius + 1):
            for dx in range(center_x - refine_radius, center_x + refine_radius + 1):
                score = iou_binary(shift2d_zero(moving, dy, dx), fixed)
                if score > best_score:
                    best_dy, best_dx, best_score = dy, dx, score
    return best_dy, best_dx, float(best_score)


def _starts(length: int, patch: int, stride: int) -> list[int]:
    if patch > length:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    final = length - patch
    if starts[-1] != final:
        starts.append(final)
    return starts


def estimate_patch_moves(
    moving_binary: np.ndarray,
    fixed_binary: np.ndarray,
    patch_size: tuple[int, int] = (500, 500),
    stride: tuple[int, int] = (500, 500),
    tissue_min_fraction: float = 0.01,
    refine_radius: int = 6,
) -> dict[str, np.ndarray]:
    """Estimate local translations over a regular, edge-covering patch grid."""
    moving_binary = np.asarray(moving_binary, dtype=bool)
    fixed_binary = np.asarray(fixed_binary, dtype=bool)
    if moving_binary.ndim != 2 or moving_binary.shape != fixed_binary.shape:
        raise ValueError("moving and fixed masks must be 2-D arrays with equal shapes")
    patch_h = min(int(patch_size[0]), moving_binary.shape[0])
    patch_w = min(int(patch_size[1]), moving_binary.shape[1])
    stride_h, stride_w = map(int, stride)
    if min(patch_h, patch_w, stride_h, stride_w) <= 0:
        raise ValueError("patch size and stride must be positive")
    if not 0 <= tissue_min_fraction <= 1:
        raise ValueError("tissue_min_fraction must be between 0 and 1")

    records: list[tuple[int, int, float, float, float, float, float]] = []
    for y0 in _starts(moving_binary.shape[0], patch_h, stride_h):
        for x0 in _starts(moving_binary.shape[1], patch_w, stride_w):
            moving_patch = moving_binary[y0 : y0 + patch_h, x0 : x0 + patch_w]
            fixed_patch = fixed_binary[y0 : y0 + patch_h, x0 : x0 + patch_w]
            if moving_patch.mean() < tissue_min_fraction or fixed_patch.mean() < tissue_min_fraction:
                continue
            estimate = best_translation_by_iou(moving_patch, fixed_patch, refine_radius)
            if estimate is not None:
                dy, dx, score = estimate
                records.append((y0, x0, y0 + patch_h / 2, x0 + patch_w / 2, dy, dx, score))

    names = ("y0", "x0", "cy", "cx", "dy", "dx", "score")
    dtypes = (np.int32, np.int32, np.float64, np.float64, np.float64, np.float64, np.float64)
    result = {
        name: np.asarray([record[index] for record in records], dtype=dtypes[index])
        for index, name in enumerate(names)
    }
    result["patch_size"] = np.asarray((patch_h, patch_w), dtype=np.int32)
    result["stride"] = np.asarray((stride_h, stride_w), dtype=np.int32)
    return result


def _interpolate_component(
    points: np.ndarray,
    values: np.ndarray,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    method: str,
) -> np.ndarray:
    if len(values) == 1:
        return np.full(grid_y.shape, values[0], dtype=np.float64)
    methods = [method] + [item for item in ("linear", "nearest") if item != method]
    for candidate in methods:
        try:
            result = griddata(points, values, (grid_y, grid_x), method=candidate, fill_value=np.nan)
        except (ValueError, RuntimeError):
            continue
        if np.isfinite(result).any():
            if np.isnan(result).any():
                nearest = griddata(points, values, (grid_y, grid_x), method="nearest")
                result = np.where(np.isfinite(result), result, nearest)
            return result
    raise ValueError("Unable to interpolate the displacement field")


def build_coarse_displacement_grid(
    shape: tuple[int, int],
    cy: np.ndarray,
    cx: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
    grid_step: int = 64,
    interpolation: str = "cubic",
    smooth_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Interpolate sparse shifts into a robust, smooth coarse field."""
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    if interpolation not in {"nearest", "linear", "cubic"}:
        raise ValueError("interpolation must be nearest, linear, or cubic")
    arrays = [np.asarray(item, dtype=np.float64) for item in (cy, cx, dy, dx)]
    if not arrays[0].size or len({item.size for item in arrays}) != 1:
        raise ValueError("At least one equally-sized patch translation is required")
    ys = np.arange(0, shape[0], grid_step, dtype=np.float64)
    xs = np.arange(0, shape[1], grid_step, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    points = np.column_stack((arrays[0], arrays[1]))
    dy_grid = _interpolate_component(points, arrays[2], grid_y, grid_x, interpolation)
    dx_grid = _interpolate_component(points, arrays[3], grid_y, grid_x, interpolation)
    if smooth_sigma > 0:
        dy_grid = gaussian_filter(dy_grid, float(smooth_sigma))
        dx_grid = gaussian_filter(dx_grid, float(smooth_sigma))
    return dy_grid.astype(np.float32), dx_grid.astype(np.float32), int(grid_step)


def _dense_field_crop(
    dy_grid: np.ndarray,
    dx_grid: np.ndarray,
    grid_step: int,
    y0: int,
    x0: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(
        np.arange(y0, y0 + shape[0], dtype=np.float32),
        np.arange(x0, x0 + shape[1], dtype=np.float32),
        indexing="ij",
    )
    y_index = np.clip(yy / grid_step, 0, dy_grid.shape[0] - 1)
    x_index = np.clip(xx / grid_step, 0, dy_grid.shape[1] - 1)
    return (
        map_coordinates(dy_grid, (y_index, x_index), order=1, mode="nearest"),
        map_coordinates(dx_grid, (y_index, x_index), order=1, mode="nearest"),
    )


def warp_labels(
    labels: np.ndarray,
    dy_grid: np.ndarray,
    dx_grid: np.ndarray,
    grid_step: int,
    tile_size: tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """Inverse-warp labels tile by tile while preserving the original IDs."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2-D image")
    out = np.zeros_like(labels)
    tile_h, tile_w = tile_size
    for y0 in range(0, labels.shape[0], tile_h):
        for x0 in range(0, labels.shape[1], tile_w):
            height = min(tile_h, labels.shape[0] - y0)
            width = min(tile_w, labels.shape[1] - x0)
            dy, dx = _dense_field_crop(dy_grid, dx_grid, grid_step, y0, x0, (height, width))
            yy, xx = np.meshgrid(
                np.arange(y0, y0 + height, dtype=np.float32),
                np.arange(x0, x0 + width, dtype=np.float32),
                indexing="ij",
            )
            warped = map_coordinates(
                labels, (yy - dy, xx - dx), order=0, mode="constant", cval=0
            )
            out[y0 : y0 + height, x0 : x0 + width] = warped.astype(labels.dtype, copy=False)
    return out


def register_masks(
    moving_labels: np.ndarray,
    fixed_labels: np.ndarray,
    *,
    patch_size: tuple[int, int] = (500, 500),
    stride: tuple[int, int] = (500, 500),
    tissue_min_fraction: float = 0.01,
    refine_radius: int = 6,
    grid_step: int = 64,
    interpolation: str = "cubic",
    smooth_sigma: float = 2.0,
) -> RegistrationResult:
    """Run CIRNE field estimation on two instance-label images."""
    moving_labels, fixed_labels = np.asarray(moving_labels), np.asarray(fixed_labels)
    if moving_labels.shape != fixed_labels.shape or moving_labels.ndim != 2:
        raise ValueError("moving and fixed labels must be 2-D arrays with equal shapes")
    moves = estimate_patch_moves(
        moving_labels > 0,
        fixed_labels > 0,
        patch_size,
        stride,
        tissue_min_fraction,
        refine_radius,
    )
    if moves["dy"].size == 0:
        raise ValueError("No eligible foreground patches were found; lower tissue_min_fraction")
    dy_grid, dx_grid, step = build_coarse_displacement_grid(
        moving_labels.shape,
        moves["cy"],
        moves["cx"],
        moves["dy"],
        moves["dx"],
        grid_step,
        interpolation,
        smooth_sigma,
    )
    return RegistrationResult(moves, dy_grid, dx_grid, step, moving_labels.shape)
