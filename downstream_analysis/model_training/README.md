# Downstream segmentation-model training

CIRNE registration does not use a neural network. This separate downstream
workflow fine-tunes CellposeSAM to generate masks that may be used as CIRNE
inputs. It is not installed with the `cirne` package.

## Data layout

Each dataset below `--data-root` must contain paired image and mask files with
the same stem:

```text
data-root/
  dataset_01/
    train/
      images/sample_001.tiff
      masks/sample_001.npy
    test/
      images/sample_101.tiff
      masks/sample_101.npy
```

TIFF masks are also accepted. The public training entry point discovers all
datasets matching this structure and records its full configuration.

## Install and run

```bash
python -m pip install -r downstream_analysis/model_training/requirements.txt

python downstream_analysis/model_training/train_cellposesam.py \
  --data-root /path/to/patches \
  --output-dir /path/to/training-output \
  --model-name CIRNE_full_whole_cell \
  --pretrained-model cpsam \
  --epochs 300 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --weight-decay 0.1 \
  --save-every 10
```

The internal reference workflow used 13 datasets, 5,200 training patches,
1,300 validation patches, and 512 × 512-pixel patches. These counts are
validated with optional `--expected-train-count` and
`--expected-validation-count` arguments.

For Slurm, edit paths and resources in [`train.slurm`](train.slurm), then run
`sbatch downstream_analysis/model_training/train.slurm`.

## Checkpoints and model hosting

Do not commit model checkpoints to Git. Pass a downloaded checkpoint to
`--pretrained-model`, or use `cpsam` and train from the Cellpose base model.
Release checkpoints can be hosted separately on Hugging Face and downloaded by
users before inference.
