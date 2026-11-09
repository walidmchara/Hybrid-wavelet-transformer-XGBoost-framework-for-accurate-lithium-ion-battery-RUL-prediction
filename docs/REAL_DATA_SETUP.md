# NASA PCoE and CALCE CS2 — real-data setup

These steps use the original archives supplied by the repository owner. The source data and derived measurements are **not** committed to Git. Run the converter locally after placing the ZIP files under `data/raw/`.

## Original archives checked

- **NASA** `5.+Battery+Data+Set.zip` (nested archives; MATLAB `.mat` cells). Relevant cells: `B0005`, `B0006`, `B0007`, `B0018`, `B0029`, `B0030`, `B0031`, `B0032`, `B0053`, `B0054`, `B0055`, `B0056`.
- **CALCE** `CS2_35.zip`, `CS2_36.zip`, `CS2_37.zip`, `CS2_38.zip`: respectively **25, 26, 27, 28 XLSX workbooks**. The `Channel_...` worksheets provide measurement samples, `Cycle_Index`, cumulative `Discharge_Capacity(Ah)` and current/voltage. Most examined sheets do not contain a recorded temperature channel.

Sources: [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), [CALCE/University of Maryland](https://calce.umd.edu/data). Download only from the dataset maintainers and follow their terms.

## Install and convert

Run from the repository root (Python 3.10+):

```bash
pip install -r requirements.txt
mkdir -p data/raw
# Copy all five user-supplied original ZIPs into data/raw/.
python scripts/prepare_battery_data.py \
  --nasa 'data/raw/5.+Battery+Data+Set.zip' \
  --calce data/raw/CS2_35.zip data/raw/CS2_36.zip data/raw/CS2_37.zip data/raw/CS2_38.zip \
  --out data/processed \
  --extract-nasa-mat-to data
```

PowerShell: use the same command on a single line (without the backslashes). To test only one workbook per CALCE cell, add `--max-calce-workbooks-per-cell 1`; this creates **partial data for checking the importer**, not a complete dataset. Full processing can be slow because all 106 XLSX workbooks must be parsed.

Outputs: `data/processed/nasa/B0005.csv`, ...; `data/processed/calce/CS2_35.csv`, ...; and `data/processed/quality_report.json` containing source ZIP SHA-256 digests, counts and data-quality flags.

## Scientific precautions

- NASA contains discharge-cycle capacity and voltage/current/temperature traces. The converter computes *observed-first-cycle SOH* `target = capacity_ah / first_valid_capacity_ah` and includes a separate nominal-2 Ah SOH column. Select and document the paper-aligned definition before comparing published values.
- CALCE capacity is **cumulative within each workbook**. Per-cycle capacity is computed from `max(Discharge_Capacity) - min(Discharge_Capacity)` *within the same workbook and `Cycle_Index`*. Never use the cumulative maximum as a cycle capacity. Boundary cycles may be incomplete; `first_or_last_cycle_in_file` marks these for review.
- CALCE `temperature_mean` is left missing rather than fabricated. Use explicit sensor columns with recorded values; do not replace missing temperature with zero.
- `remaining_observed_cycles` is a countdown to the **end of the available record**, not a verified end-of-life RUL. Do not report it as the published RUL label without confirming EOL criteria and censoring.
- Train/validation/test splits must be made by battery ID for cross-cell experiments. Never concatenate cells and then use a row-wise split. Existing single-sequence prototype training scripts are only suitable for individual-cell exploratory checks; they do not establish cross-cell generalization.
- Never use `capacity_ah`, `soh_first_observed`, `soh_nominal`, `remaining_observed_cycles` or any post-target measurement as predictors of the same-cycle target. Use only observations available at forecast time; for RUL, validate the target's definition and observation horizon.
- A passing import or smoke test confirms program execution, **not** exact reproduction of any published experiment.

## Recommended data quality checks

Review `quality_report.json`, raw-versus-processed capacity trends, missing values, and per-cell sample counts. Audit duplicate overlapping CALCE workbooks, incomplete boundary cycles, changes in loading protocol and abrupt capacity recovery before accepting any evaluation data. Large or raw datasets remain in ignored `data/` paths.

## Use the existing WTX-CBO entry point

The original NASA `.mat` files are extracted to `data/` for the existing NASA experiment:

```bash
python main.py --nasa-mat data/B0005.mat data/B0006.mat data/B0007.mat data/B0018.mat --test-battery B0018 --epochs 2 --output-dir outputs/nasa_b0018_test
```

A **CALCE exploratory example** (not an EOL/RUL reproduction):

```bash
python main.py --csv data/processed/calce/CS2_38.csv --target-col remaining_observed_cycles --feature-cols voltage_mean voltage_std current_mean current_std discharge_duration_s --epochs 2 --output-dir outputs/calce_cs2_38_exploratory
```

The CALCE example's target is time to the *last observed cycle*, not time to a validated EOL. Published RUL comparisons require a separately verified stopping criterion and target labelling rule. Features must be explicitly selected to prevent accidental leakage from the label-related columns.
