# Downstream analysis

This directory contains research workflows that use, evaluate, or prepare data
for CIRNE but are not part of the installable registration package.

- `model_training/`: optional CellposeSAM training used to generate instance
  masks. It has its own dependencies and is not required by CIRNE.
- `PROVENANCE.md`: mapping from the internal research workspace to the curated
  public implementation.

Code under this directory is intentionally excluded from the `cirne` Python
package. Installing CIRNE only installs modules from `src/cirne`.

