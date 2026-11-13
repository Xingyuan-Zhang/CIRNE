# Research-code provenance

This public package was reconstructed from the CIRNE research workspace. The
mapping below records the source of each public component without exposing
cluster-specific paths or large data products.

| Public component | Original research component | Change for release |
|---|---|---|
| `src/cirne/core.py` | `script/cirne.py` | Split into importable API; validation and sparse-point interpolation fallback added |
| `src/cirne/cli.py` | CLI in `script/cirne.py` | Added raw-image warping, generalized NPY/TIFF inputs, and made intermediate outputs optional |
| `downstream_analysis/model_training/train_cellposesam.py` | `script/train_cirne_full_whole_cell.py` | Archived as downstream analysis; removed hard-coded HPC paths and inferred-resume state |
| `downstream_analysis/model_training/train.slurm` | `script/train_cirne_he_to_fixed_dapi.slurm` | Archived as downstream analysis; replaced site-specific modules and paths |
| workflow description | `cirne_workflow_figure/README.md` | Condensed into the repository overview |

## Deliberately excluded

- patient or sample data;
- large microscopy images and intermediate registration arrays;
- trained model weights and Cellpose checkpoints;
- absolute HPC paths;
- exploratory scripts tied to unpublished local directory layouts.

The larger workspace also contains dataset-specific evaluation and figure
scripts. Those should only be published after their input contracts, sample
metadata, and redistribution permissions are reviewed.
