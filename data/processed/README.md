# Processed battery measurements

The 12 NASA and four CALCE cell-level CSVs in this folder are **real observational data derived from the source archives**, not fabricated samples. See `manifest.json` for the exact dataset IDs, row counts and original ZIP SHA-256 fingerprints.

- NASA: `nasa/B0005.csv`, `B0006.csv`, `B0007.csv`, `B0018.csv`, `B0029.csv`, `B0030.csv`, `B0031.csv`, `B0032.csv`, `B0053.csv`, `B0054.csv`, `B0055.csv`, `B0056.csv`.
- CALCE: `calce/CS2_35.csv`, `CS2_36.csv`, `CS2_37.csv`, `CS2_38.csv` (all 106 source XLSX workbooks processed).

See [real-data setup](../../docs/REAL_DATA_SETUP.md) for input/target definitions, data leakage warnings, source citations and reprocessing commands. Original large ZIP/MAT/XLSX archives are not part of Git history.
