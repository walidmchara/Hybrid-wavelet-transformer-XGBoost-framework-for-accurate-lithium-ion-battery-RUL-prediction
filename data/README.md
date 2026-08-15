# Data directory

Place the battery datasets here.

## NASA PCoE

Expected files for the experiment described in the paper:

- `B0005.mat`
- `B0006.mat`
- `B0007.mat`
- `B0018.mat`

The paper uses B0005, B0006 and B0007 for training/validation and reserves B0018 for testing.

## CALCE

You can also use pre-aggregated cycle-level CSV files. A CSV should contain one row per cycle and numeric feature columns such as:

- voltage
- current
- temperature
- capacity
- cycle index

If a column named `rul` is present, pass `--target-col rul`. Otherwise the code generates an end-of-series remaining-cycle target.

> Large datasets should not be committed to GitHub. Keep this folder locally and commit only this README.
