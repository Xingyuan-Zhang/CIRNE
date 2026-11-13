# CIRNE

**CIRNE** is a lightweight, instance-mask-guided nonrigid registration tool for
spatial imaging. It estimates local translations from foreground masks, builds
a smooth deformation field, and uses it to warp the corresponding raw image.
CIRNE is an independent registration project.

## Method overview

![Figure 1B. CIRNE registration workflow](docs/assets/figure1b_cirne_workflow.png)

**Figure 1B. CIRNE registration workflow.** DAPI and H&E are first brought into
global alignment using an upstream affine-registration step. CIRNE then
estimates patch-wise translations by maximizing nuclear-mask IoU, interpolates
and smooths these sparse shifts into a dense displacement field, and applies an
inverse warp to the moving image or instance-label mask. The released Python
package implements the highlighted CIRNE core (steps 3–5); image segmentation
and global affine alignment are upstream preprocessing and are not installed as
part of `cirne`.

## How it works

1. Divide the moving and fixed masks into spatial patches.
2. Estimate each eligible patch's integer translation by FFT correlation and
   local IoU refinement.
3. Interpolate the patch translations into a smooth displacement field.
4. Inverse-warp the raw moving image using bilinear interpolation. Instance
   labels can optionally be warped with nearest-neighbour sampling.

CIRNE estimates geometry from binary foreground occupancy; it does not perform
one-to-one matching between cell IDs in the two images.

## Installation

```bash
git clone https://github.com/Xingyuan-Zhang/CIRNE.git
cd CIRNE
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e '.[test]'
pytest
```

## Quick start

Provide two 2-D nuclei label masks and the raw moving image associated with the
moving nuclei. All three inputs must have identical spatial dimensions.
Background in the masks is zero and positive integers are instance IDs. The raw
image may be 2-D grayscale or 3-D channel-last. NPY and TIFF are supported.

```bash
cirne-register \
  --moving-nuclei moving_nuclei.npy \
  --fixed-nuclei fixed_nuclei.npy \
  --moving-image moving_raw_image.tif \
  --output warped_image.tif \
  --patch-size 500 \
  --stride 500
```

By default, the command writes exactly one file:

- `warped_image.tif`: the registered raw moving image.

Intermediate outputs are opt-in:

```bash
cirne-register \
  --moving-nuclei moving_nuclei.npy \
  --fixed-nuclei fixed_nuclei.npy \
  --moving-image moving_raw_image.tif \
  --output warped_image.tif \
  --save-field \
  --save-patch-moves \
  --save-summary \
  --save-warped-nuclei
```

The Python API exposes the same workflow:

```python
from cirne import register_masks, warp_image

result = register_masks(moving_nuclei, fixed_nuclei, patch_size=(500, 500))
warped = warp_image(
    moving_raw_image,
    result.dy_grid,
    result.dx_grid,
    result.grid_step,
)
```

See [`examples/synthetic_registration.py`](examples/synthetic_registration.py)
for a complete runnable example.

## Parameter guidance

- `patch-size`: should contain enough foreground structure to determine a
  unique translation. Start with 500 pixels for the original workflow.
- `stride`: use the patch size for non-overlapping estimation; decrease it for
  denser estimates at higher compute cost.
- `refine-radius`: local IoU search radius around the correlation estimate.
- `grid-step`: spacing of the coarse deformation grid. Smaller values produce
  a denser field but use more memory.
- `smooth-sigma`: Gaussian smoothing on the coarse grid.

## Package scope

The installable `cirne` package contains only the model-free CIRNE registration
algorithm, label-image I/O, and command-line interface. It does not depend on
Cellpose, PyTorch, or any segmentation-model weights.

Segmentation-model training is not part of CIRNE. For research reproducibility,
the associated CellposeSAM training scripts are archived separately under
[`downstream_analysis/model_training`](downstream_analysis/model_training) and
are excluded from the Python package and its dependencies.

## Reproducibility and downstream analysis

The correspondence between the original research workspace and this public
repository is documented in
[`downstream_analysis/PROVENANCE.md`](downstream_analysis/PROVENANCE.md). Large
raw images, intermediate arrays, and model checkpoints are excluded.

## Citation

If CIRNE contributes to your work, please cite this software repository using
the metadata in [`CITATION.cff`](CITATION.cff).

## License

MIT License. See [`LICENSE`](LICENSE).
