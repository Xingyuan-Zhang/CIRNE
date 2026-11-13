"""Input/output helpers for label images and CIRNE results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


def load_labels(path: str | Path) -> np.ndarray:
    """Load a 2-D instance-label image from NPY, TIFF, or TIFF-compatible files."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        labels = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        labels = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported label format: {path.suffix}. Use .npy, .tif, or .tiff")
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2-D label image, got shape {labels.shape} from {path}")
    if not (np.issubdtype(labels.dtype, np.integer) or labels.dtype == bool):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.rint(labels)):
            raise ValueError(f"Label image must contain finite integer-valued IDs: {path}")
        labels = np.rint(labels).astype(np.int64)
    if np.any(labels < 0):
        raise ValueError(f"Label IDs must be non-negative: {path}")
    return labels


def save_labels(path: str | Path, labels: np.ndarray) -> None:
    """Save an instance-label image as NPY or TIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, labels, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(path, labels)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")

