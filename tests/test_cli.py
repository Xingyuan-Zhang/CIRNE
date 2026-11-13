from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from cirne import shift2d_zero
from cirne.cli import main


def test_cli_default_writes_only_warped_raw_image() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        moving = np.zeros((80, 96), dtype=np.uint16)
        moving[12:30, 15:38] = 1
        moving[42:68, 55:82] = 2
        fixed = shift2d_zero(moving, 3, -5)
        raw = np.stack((moving > 0, moving == 1, moving == 2), axis=-1).astype(np.uint8) * 255
        np.save(root / "moving_nuclei.npy", moving)
        np.save(root / "fixed_nuclei.npy", fixed)
        np.save(root / "moving_raw.npy", raw)

        output = root / "warped_raw.npy"
        assert main(
            [
                "--moving-nuclei",
                str(root / "moving_nuclei.npy"),
                "--fixed-nuclei",
                str(root / "fixed_nuclei.npy"),
                "--moving-image",
                str(root / "moving_raw.npy"),
                "--output",
                str(output),
                "--patch-size",
                "80",
                "--stride",
                "80",
                "--grid-step",
                "20",
                "--smooth-sigma",
                "0",
                "--image-interpolation-order",
                "0",
            ]
        ) == 0
        assert output.is_file()
        outputs = {path.name for path in root.iterdir()} - {
            "moving_nuclei.npy",
            "fixed_nuclei.npy",
            "moving_raw.npy",
        }
        assert outputs == {"warped_raw.npy"}
