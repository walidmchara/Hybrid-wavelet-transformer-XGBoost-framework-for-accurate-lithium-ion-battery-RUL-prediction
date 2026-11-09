# NASA and CALCE battery datasets

The repository includes measured discharge-cycle aggregates derived from the NASA PCoE and CALCE CS2 battery-aging archives.

## Data organization

```text
data/
  processed/
    nasa/    B0005.csv, B0006.csv, B0007.csv, B0018.csv, ...
    calce/   CS2_35.csv, CS2_36.csv, CS2_37.csv, CS2_38.csv
    manifest.json
```

The NASA directory covers 12 battery cells; the CALCE directory covers four CS2 cells across 106 source workbooks. The manifest records input archive checksums and per-file provenance.

## RUL benchmark

The paper's NASA experiment uses B0005, B0006 and B0007 for development and B0018 as the held-out cell. The `--nasa-mat` implementation consumes original MATLAB discharge traces, which are kept separately from the compact CSVs. The procedure for extracting the source data is in [../docs/REAL_DATA_SETUP.md](../docs/REAL_DATA_SETUP.md).

The processed CSV column `remaining_observed_cycles` is an observed-record countdown; it is **not** an independently established end-of-life (EOL) RUL label. A valid RUL benchmark requires a documented EOL criterion, cycle eligibility rules and censoring policy. An exploratory check on CALCE data can be run with:

```bash
python main.py --csv data/processed/calce/CS2_35.csv --target-col remaining_observed_cycles --epochs 2 --output-dir outputs/calce_quickcheck
```

Sources: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE/UMD](https://calce.umd.edu/data). Derived measurements should be distinguished from their original source files in any reported results.
