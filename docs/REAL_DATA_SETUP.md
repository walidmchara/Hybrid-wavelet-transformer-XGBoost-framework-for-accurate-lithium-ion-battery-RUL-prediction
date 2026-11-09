# Real datasets installed: NASA PCoE and CALCE CS2

## Files checked into the repository
- `data/processed/nasa/B0005.csv` through the 12 selected batteries listed in `data/processed/manifest.json` (1,157 discharge-cycle rows across 12 cells).
- `data/processed/calce/CS2_35.csv`, `CS2_36.csv`, `CS2_37.csv`, `CS2_38.csv` (3,921 cycle-level rows across **all 106 CALCE XLSX workbooks**).
- `data/processed/manifest.json`: input archive SHA-256 fingerprints, cell IDs, output record counts.
- `scripts/prepare_battery_data.py`: the code used to regenerate derived CSVs.

The original `.mat` and Excel/ZIP archives are **not** tracked because they are very large; derived CSVs are tracked. Source links: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and [CALCE/UMD](https://calce.umd.edu/data). NASA asks researchers to acknowledge data donors; CALCE asks publications to cite the relevant experimental studies. Verify conditions for subsequent redistribution.

## Derived columns and warnings

Each CSV has `cycle`, `time`, `capacity_ah`, `soh_first_observed`, `target`, `cell_id`, diagnostic statistics and `remaining_observed_cycles`. SOH here is relative to the **first observed** discharge capacity, not necessarily the published nominal rating. `remaining_observed_cycles` counts records until *the end of the observed file* and is **not** confirmed EOL-based RUL.

NASA: one row per valid discharge cycle. Some cells have `target > 1` because the first measured capacity is lower than later measurements; do not clip these silently. CALCE: discharge capacity is cumulative **within each workbook**; conversion uses capacity maximum minus minimum per valid cycle, corrects identical-time duplicates, and flags workbook boundary cycles. CALCE measurement sheets lack an observed temperature channel, so `temperature_mean` is missing.

**Avoid target leakage.** Do not feed the same-cycle `capacity_ah`, `soh_first_observed`, `soh_nominal`, `target`, or `remaining_observed_cycles` as predictor features for an SOH model. The current reconstruction's `train.py` appends *past* target values to historical windows; this is an autoregressive research assumption, not an independently validated paper implementation. Split by cell for cross-battery benchmarks; the default one-CSV chronology is a pipeline test only.

## Running the repository on the committed data

The WTX-CBO repository uses `main.py` (not `train.py`). Its experimental RUL definition must be audited separately from the capacity-based SOH target:

```bash
python -m pip install -r requirements.txt
# Illustrative CALCE observed-end countdown (NOT confirmed physical EOL RUL)
python main.py --csv data/processed/calce/CS2_35.csv --target-col remaining_observed_cycles --epochs 2 --output-dir outputs/calce_quickcheck
```

For the paper's stated NASA `B0005/B0006/B0007` training and `B0018` test setting, the **original MATLAB files** are still needed for the `--nasa-mat` path; they are not stored in Git because of their size. Extract them locally from the NASA ZIP with `scripts/prepare_battery_data.py --extract-nasa-mat-to data/raw/nasa` and specify the resulting paths to `--nasa-mat`.

A capacity-based SOH ratio and an end-of-record countdown are not equivalent to the paper's EOL-RUL definition. Do not report outputs from these convenience CSV files as published paper replication. Cross-battery and cross-dataset experiments need independent holdout protocols.

## Regenerating from original archives

Put the downloaded `5.+Battery+Data+Set.zip` and `CS2_35.zip`–`CS2_38.zip` into `data/raw/`, then:

```bash
python scripts/prepare_battery_data.py --nasa 'data/raw/5.+Battery+Data+Set.zip' --calce data/raw/CS2_35.zip data/raw/CS2_36.zip data/raw/CS2_37.zip data/raw/CS2_38.zip --out data/processed --extract-nasa-mat-to data/raw/nasa
```

CSV data are derived from real measurements, **not** synthetic demonstration data. They have not been used here to verify published paper-level accuracy.
