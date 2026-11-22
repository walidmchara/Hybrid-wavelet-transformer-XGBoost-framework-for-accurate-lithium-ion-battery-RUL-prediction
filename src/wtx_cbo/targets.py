"""Explicit, auditable EOL/RUL target construction for cycle-level battery data.

A dataset ending is NOT an observed EOL event. Censored cells must not be
silently relabeled as having zero remaining useful life.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLUMNS = {"rul", "remaining_observed_cycles", "target", "soh",
                 "capacity", "discharge_capacity", "capacity_ah",
                 "remaining_useful_life", "eol_cycle", "cycle_life"}

def safe_predictor_columns(frame: pd.DataFrame, requested=None):
    """Only time-available numeric covariates; exclude direct label proxies."""
    columns = list(requested) if requested is not None else list(frame.select_dtypes(include=np.number))
    if requested is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Requested predictors not present in the input: {missing}")
    forbidden = [c for c in columns if c.lower() in LABEL_COLUMNS or
                 any(token in c.lower() for token in ("capacity", "soh", "rul", "remaining_", "eol_"))]
    if forbidden and requested is not None:
        raise ValueError(f"Target leakage risk: requested predictors {forbidden}")
    safe = [c for c in columns if c not in forbidden and c != "cycle"]
    if "cycle" in columns:
        safe.insert(0, "cycle")
    if not safe:
        raise ValueError("No safe numeric predictors left")
    return safe

def eol_rul_from_capacity(frame: pd.DataFrame, *, threshold_ah: float,
                          capacity_col: str = "capacity", cycle_col: str = "cycle",
                          min_consecutive: int = 1) -> pd.DataFrame:
    """Label only an *observed* first capacity EOL threshold crossing.

    Returned rows end at observed EOL. RUL is the number of discharge events
    before the observed crossing, rather than remaining dataset rows.
    Threshold is an explicit scientific choice, never inferred from test data.
    """
    if not np.isfinite(threshold_ah) or threshold_ah <= 0:
        raise ValueError("threshold_ah must be a positive, finite absolute capacity")
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")
    if capacity_col not in frame or cycle_col not in frame:
        raise ValueError(f"Expected {capacity_col!r} and {cycle_col!r} columns")
    d = frame.sort_values(cycle_col).reset_index(drop=True).copy()
    if d[cycle_col].duplicated().any():
        raise ValueError("Duplicate discharge cycle IDs")
    c = pd.to_numeric(d[capacity_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("Invalid/missing capacity. Audit raw data before RUL labeling.")
    below = c <= threshold_ah
    event = next((i for i in range(len(c)-min_consecutive+1)
                  if below[i:i+min_consecutive].all()), None)
    if event is None:
        raise ValueError("CENSORED: no observed capacity EOL threshold crossing. "
                         "Do not treat last observed cycle as EOL.")
    if event == 0:
        raise ValueError("EOL at first discharge; no degradation trajectory")
    d = d.iloc[:event+1].copy()
    d["rul"] = np.arange(event, -1, -1, dtype=float)
    d.attrs["eol_event_index"] = event
    d.attrs["eol_threshold_ah"] = float(threshold_ah)
    return d
