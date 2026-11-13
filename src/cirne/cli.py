"""Command-line interface for CIRNE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import mask_metrics, register_masks, warp_image, warp_labels
from .io import load_image, load_labels, save_image, save_labels, write_json


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _artifact_path(output: Path, label: str, suffix: str) -> Path:
    return output.with_name(f"{output.stem}.{label}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cirne-register",
        description=(
            "Estimate registration from moving/fixed nuclei masks and warp the corresponding raw image."
        ),
    )
    parser.add_argument(
        "--moving-nuclei",
        "--moving",
        dest="moving_nuclei",
        required=True,
        help="Moving 2-D nuclei labels (.npy/.tif/.tiff)",
    )
    parser.add_argument(
        "--fixed-nuclei",
        "--fixed",
        dest="fixed_nuclei",
        required=True,
        help="Fixed 2-D nuclei labels (.npy/.tif/.tiff)",
    )
    parser.add_argument(
        "--moving-image",
        required=True,
        help="Raw moving grayscale or channel-last image (.npy/.tif/.tiff)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Warped raw image path (.npy/.tif/.tiff). This is the only default output.",
    )
    parser.add_argument("--patch-size", type=_positive_int, default=500)
    parser.add_argument("--stride", type=_positive_int, default=500)
    parser.add_argument("--tissue-min-fraction", type=float, default=0.01)
    parser.add_argument("--refine-radius", type=int, default=6)
    parser.add_argument("--grid-step", type=_positive_int, default=64)
    parser.add_argument("--interpolation", choices=("nearest", "linear", "cubic"), default="cubic")
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--tile-size", type=_positive_int, default=1024)
    parser.add_argument(
        "--image-interpolation-order",
        type=int,
        choices=(0, 1),
        default=1,
        help="Raw-image interpolation: 0=nearest, 1=bilinear (default: 1)",
    )
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument(
        "--save-field", action="store_true", help="Also save the coarse displacement field"
    )
    parser.add_argument(
        "--save-patch-moves", action="store_true", help="Also save patch-wise translations"
    )
    parser.add_argument(
        "--save-summary", action="store_true", help="Also save parameters and mask metrics"
    )
    parser.add_argument(
        "--save-warped-nuclei", action="store_true", help="Also save the warped moving nuclei"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if output.suffix.lower() not in {".npy", ".tif", ".tiff"}:
        raise ValueError("--output must end in .npy, .tif, or .tiff")

    moving_nuclei = load_labels(args.moving_nuclei)
    fixed_nuclei = load_labels(args.fixed_nuclei)
    moving_image = load_image(args.moving_image)
    if moving_image.shape[:2] != moving_nuclei.shape:
        raise ValueError(
            "Raw moving image and moving nuclei must have identical spatial dimensions: "
            f"{moving_image.shape[:2]} vs {moving_nuclei.shape}"
        )

    result = register_masks(
        moving_nuclei,
        fixed_nuclei,
        patch_size=(args.patch_size, args.patch_size),
        stride=(args.stride, args.stride),
        tissue_min_fraction=args.tissue_min_fraction,
        refine_radius=args.refine_radius,
        grid_step=args.grid_step,
        interpolation=args.interpolation,
        smooth_sigma=args.smooth_sigma,
    )
    warped_image = warp_image(
        moving_image,
        result.dy_grid,
        result.dx_grid,
        result.grid_step,
        tile_size=(args.tile_size, args.tile_size),
        interpolation_order=args.image_interpolation_order,
        fill_value=args.fill_value,
    )
    save_image(output, warped_image)

    if args.save_field:
        np.savez_compressed(
            _artifact_path(output, "field", ".npz"),
            dy=result.dy_grid,
            dx=result.dx_grid,
            grid_step=np.asarray([result.grid_step], dtype=np.int32),
            shape=np.asarray(result.shape, dtype=np.int64),
        )
    if args.save_patch_moves:
        np.savez_compressed(_artifact_path(output, "patch_moves", ".npz"), **result.patch_moves)

    warped_nuclei = None
    if args.save_warped_nuclei or args.save_summary:
        warped_nuclei = warp_labels(
            moving_nuclei,
            result.dy_grid,
            result.dx_grid,
            result.grid_step,
            (args.tile_size, args.tile_size),
        )
    if args.save_warped_nuclei:
        nuclei_suffix = ".npy" if output.suffix.lower() == ".npy" else ".tif"
        save_labels(_artifact_path(output, "nuclei", nuclei_suffix), warped_nuclei)
    if args.save_summary:
        summary = {
            "moving_nuclei": str(Path(args.moving_nuclei).resolve()),
            "fixed_nuclei": str(Path(args.fixed_nuclei).resolve()),
            "moving_image": str(Path(args.moving_image).resolve()),
            "output": str(output.resolve()),
            "shape": list(result.shape),
            "patches_used": int(result.patch_moves["dy"].size),
            "metrics_before": mask_metrics(moving_nuclei > 0, fixed_nuclei > 0),
            "metrics_after": mask_metrics(warped_nuclei > 0, fixed_nuclei > 0),
            "parameters": vars(args),
        }
        write_json(_artifact_path(output, "summary", ".json"), summary)

    print(f"CIRNE finished using {result.patch_moves['dy'].size} eligible patches")
    print(f"Warped image: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
