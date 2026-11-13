"""Command-line interface for CIRNE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import mask_metrics, register_masks, warp_labels
from .io import load_labels, save_labels, write_json


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cirne-register",
        description="Register a moving instance-label mask to a fixed label mask.",
    )
    parser.add_argument("--moving", required=True, help="Moving .npy/.tif/.tiff instance labels")
    parser.add_argument("--fixed", required=True, help="Fixed .npy/.tif/.tiff instance labels")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--patch-size", type=_positive_int, default=500)
    parser.add_argument("--stride", type=_positive_int, default=500)
    parser.add_argument("--tissue-min-fraction", type=float, default=0.01)
    parser.add_argument("--refine-radius", type=int, default=6)
    parser.add_argument("--grid-step", type=_positive_int, default=64)
    parser.add_argument("--interpolation", choices=("nearest", "linear", "cubic"), default="cubic")
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--tile-size", type=_positive_int, default=1024)
    parser.add_argument("--warped-format", choices=("npy", "tif"), default="npy")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    moving, fixed = load_labels(args.moving), load_labels(args.fixed)
    result = register_masks(
        moving,
        fixed,
        patch_size=(args.patch_size, args.patch_size),
        stride=(args.stride, args.stride),
        tissue_min_fraction=args.tissue_min_fraction,
        refine_radius=args.refine_radius,
        grid_step=args.grid_step,
        interpolation=args.interpolation,
        smooth_sigma=args.smooth_sigma,
    )
    warped = warp_labels(
        moving, result.dy_grid, result.dx_grid, result.grid_step, (args.tile_size, args.tile_size)
    )
    np.savez_compressed(output / "patch_moves.npz", **result.patch_moves)
    np.savez_compressed(
        output / "coarse_field.npz",
        dy=result.dy_grid,
        dx=result.dx_grid,
        grid_step=np.asarray([result.grid_step], dtype=np.int32),
        shape=np.asarray(result.shape, dtype=np.int64),
    )
    save_labels(output / f"warped_labels.{args.warped_format}", warped)
    before = mask_metrics(moving > 0, fixed > 0)
    after = mask_metrics(warped > 0, fixed > 0)
    summary = {
        "moving": str(Path(args.moving).resolve()),
        "fixed": str(Path(args.fixed).resolve()),
        "shape": list(result.shape),
        "patches_used": int(result.patch_moves["dy"].size),
        "metrics_before": before,
        "metrics_after": after,
        "parameters": vars(args),
    }
    write_json(output / "summary.json", summary)
    print(f"CIRNE finished: IoU {before['iou']:.4f} -> {after['iou']:.4f}")
    print(f"Results: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

