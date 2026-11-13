#!/usr/bin/env python3
"""Portable CellposeSAM fine-tuning entry point for the CIRNE workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="CIRNE_full_whole_cell")
    parser.add_argument("--pretrained-model", default="cpsam")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--expected-train-count", type=int)
    parser.add_argument("--expected-validation-count", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def paired_paths(root: Path, split: str) -> tuple[list[Path], list[Path]]:
    image_paths = sorted(root.glob(f"*/{split}/images/*"))
    pairs: list[tuple[Path, Path]] = []
    for image in image_paths:
        mask_dir = image.parent.parent / "masks"
        candidates = [mask_dir / f"{image.stem}.npy", mask_dir / f"{image.stem}.tif", mask_dir / f"{image.stem}.tiff"]
        mask = next((candidate for candidate in candidates if candidate.exists()), None)
        if mask is None:
            raise FileNotFoundError(f"No mask found for {image}")
        pairs.append((image, mask))
    if not pairs:
        raise FileNotFoundError(f"No paired files found for split '{split}' below {root}")
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else tifffile.imread(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_images, train_masks = paired_paths(args.data_root, "train")
    validation_images, validation_masks = paired_paths(args.data_root, "test")
    for actual, expected, name in (
        (len(train_images), args.expected_train_count, "training"),
        (len(validation_images), args.expected_validation_count, "validation"),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"Expected {expected} {name} pairs, found {actual}")

    config = vars(args).copy()
    config["data_root"] = str(args.data_root.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    config["train_count"] = len(train_images)
    config["validation_count"] = len(validation_images)
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from cellpose import io, models, train

    images = [load(path) for path in train_images]
    labels = [load(path) for path in train_masks]
    validation_data = [load(path) for path in validation_images]
    validation_labels = [load(path) for path in validation_masks]
    io.logger_setup()
    model = models.CellposeModel(gpu=not args.cpu, pretrained_model=args.pretrained_model)
    model_path, train_losses, validation_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        test_data=validation_data,
        test_labels=validation_labels,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
        save_each=True,
        save_path=str(args.output_dir),
        model_name=args.model_name,
    )
    np.save(args.output_dir / "train_losses.npy", np.asarray(train_losses))
    np.save(args.output_dir / "validation_losses.npy", np.asarray(validation_losses))
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
