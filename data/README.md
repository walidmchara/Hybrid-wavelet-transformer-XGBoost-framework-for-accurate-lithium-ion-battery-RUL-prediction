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


## User-supplied NASA + CALCE source archives

The NASA outer ZIP contains nested archives with the `.mat` files. CALCE CS2_35–38 contain 25/26/27/28 `.xlsx` files; the original CALCE discharge capacity column is cumulative, so it must be differenced per cycle. See [`../docs/REAL_DATA_SETUP.md`](../docs/REAL_DATA_SETUP.md). Run `python scripts/prepare_battery_data.py --help` from the project root. Never commit the original archives or processed CSVs.

## Committed cycle-level datasets

The [processed/](processed/) directory contains real NASA and CALCE discharge-cycle CSVs and a [source manifest](processed/manifest.json). These files are derived aggregates. Full MATLAB traces and original CALCE Excel workbooks are not included in Git. Refer to [docs/REAL_DATA_SETUP.md](../docs/REAL_DATA_SETUP.md) for caveats about SOH and observed-end RUL.
