import numpy as np

from cirne import (
    best_translation_by_iou,
    build_coarse_displacement_grid,
    iou_binary,
    register_masks,
    shift2d_zero,
    warp_image,
    warp_labels,
)


def labels() -> np.ndarray:
    image = np.zeros((128, 144), dtype=np.uint16)
    image[15:35, 20:48] = 1
    image[48:80, 65:92] = 2
    image[88:115, 104:132] = 7
    return image


def test_translation_is_recovered() -> None:
    moving = labels()
    fixed = shift2d_zero(moving, 6, -9)
    dy, dx, score = best_translation_by_iou(moving, fixed)
    assert (dy, dx) == (6, -9)
    assert score == 1.0


def test_registration_improves_iou_and_preserves_ids() -> None:
    moving = labels()
    fixed = shift2d_zero(moving, -5, 8)
    result = register_masks(
        moving,
        fixed,
        patch_size=moving.shape,
        stride=moving.shape,
        grid_step=32,
        smooth_sigma=0,
    )
    warped = warp_labels(moving, result.dy_grid, result.dx_grid, result.grid_step)
    assert iou_binary(warped, fixed) > iou_binary(moving, fixed)
    assert iou_binary(warped, fixed) == 1.0
    assert set(np.unique(warped)) <= set(np.unique(moving))


def test_single_patch_builds_constant_field() -> None:
    dy_grid, dx_grid, step = build_coarse_displacement_grid(
        (100, 120),
        np.array([50.0]),
        np.array([60.0]),
        np.array([3.0]),
        np.array([-4.0]),
        grid_step=20,
        interpolation="cubic",
        smooth_sigma=0,
    )
    assert step == 20
    assert np.all(dy_grid == 3)
    assert np.all(dx_grid == -4)


def test_rgb_raw_image_is_warped_with_the_mask_field() -> None:
    moving = labels()
    fixed = shift2d_zero(moving, 4, -6)
    raw = np.stack((moving > 0, moving == 2, moving == 7), axis=-1).astype(np.uint8) * 255
    result = register_masks(
        moving,
        fixed,
        patch_size=moving.shape,
        stride=moving.shape,
        grid_step=32,
        smooth_sigma=0,
    )
    warped = warp_image(
        raw,
        result.dy_grid,
        result.dx_grid,
        result.grid_step,
        interpolation_order=0,
    )
    expected = shift2d_zero(raw[..., 0], 4, -6)
    assert warped.shape == raw.shape
    assert warped.dtype == raw.dtype
    assert np.array_equal(warped[..., 0], expected)


def test_empty_iou_is_one() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    assert iou_binary(empty, empty) == 1.0
