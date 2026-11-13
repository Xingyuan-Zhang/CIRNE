"""Run CIRNE on a small synthetic translated label image."""

from pathlib import Path

import numpy as np

from cirne import iou_binary, register_masks, shift2d_zero, warp_labels


def main() -> None:
    moving = np.zeros((256, 256), dtype=np.uint16)
    moving[35:70, 45:85] = 1
    moving[100:145, 80:125] = 2
    moving[165:215, 155:205] = 3
    fixed = shift2d_zero(moving, dy=8, dx=-11)

    result = register_masks(
        moving,
        fixed,
        patch_size=(256, 256),
        stride=(256, 256),
        smooth_sigma=0,
    )
    warped = warp_labels(moving, result.dy_grid, result.dx_grid, result.grid_step)
    print("Estimated shift:", result.patch_moves["dy"][0], result.patch_moves["dx"][0])
    print("IoU before:", iou_binary(moving, fixed))
    print("IoU after: ", iou_binary(warped, fixed))
    output = Path("synthetic_cirne_output.npy")
    np.save(output, warped)
    print("Saved", output)


if __name__ == "__main__":
    main()

