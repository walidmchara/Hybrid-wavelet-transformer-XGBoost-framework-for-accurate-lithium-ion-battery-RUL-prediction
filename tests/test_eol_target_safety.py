import numpy as np
import pandas as pd
import pytest

from wtx_cbo.targets import eol_rul_from_capacity, safe_predictor_columns
from wtx_cbo.core import load_generic_csv


def sample():
    return pd.DataFrame({"cycle": [1, 2, 3, 4, 5],
                         "capacity": [2.0, 1.9, 1.55, 1.38, 1.31],
                         "voltage_mean": [4.1, 4.0, 3.9, 3.8, 3.7],
                         "soh": [1.0, .95, .775, .69, .655]})


def test_eol_stops_at_first_observed_crossing():
    r = eol_rul_from_capacity(sample(), threshold_ah=1.4)
    assert r["cycle"].tolist() == [1, 2, 3, 4]
    assert r["rul"].tolist() == [3., 2., 1., 0.]


def test_censored_data_fails_closed():
    with pytest.raises(ValueError, match="CENSORED"):
        eol_rul_from_capacity(sample(), threshold_ah=0.8)


def test_invalid_capacity_rejected():
    frame = sample()
    frame.loc[2, "capacity"] = np.nan
    with pytest.raises(ValueError, match="capacity"):
        eol_rul_from_capacity(frame, threshold_ah=1.4)


def test_consecutive_crossing():
    frame = sample()
    r = eol_rul_from_capacity(frame, threshold_ah=1.4, min_consecutive=2)
    assert len(r) == 4


def test_safe_features_exclude_target_proxies():
    s = safe_predictor_columns(sample().assign(rul=[4, 3, 2, 1, 0],
                                                remaining_observed_cycles=[4,3,2,1,0]))
    assert s == ["cycle", "voltage_mean"]


def test_explicit_leaky_feature_rejected():
    with pytest.raises(ValueError, match="leakage"):
        safe_predictor_columns(sample(), ["voltage_mean", "capacity"])


def test_csv_requires_explicit_verified_target(tmp_path):
    path = tmp_path / "cells.csv"
    sample().to_csv(path, index=False)
    with pytest.raises(ValueError, match="Explicit"):
        load_generic_csv(path)
    sample().assign(remaining_observed_cycles=[4,3,2,1,0]).to_csv(path,index=False)
    with pytest.raises(ValueError, match="dataset-end proxy"):
        load_generic_csv(path, target_col="remaining_observed_cycles")


def test_custom_target_is_not_available_as_a_predictor(tmp_path):
    path = tmp_path / "cells.csv"
    sample().assign(days_left=[4, 3, 2, 1, 0]).to_csv(path, index=False)
    loaded = load_generic_csv(path, target_col="days_left")
    assert loaded["rul"].tolist() == [4., 3., 2., 1., 0.]
    assert "days_left" not in loaded.columns
    assert "days_left" not in safe_predictor_columns(loaded)
    with pytest.raises(ValueError, match="not present"):
        safe_predictor_columns(loaded, ["days_left"])


def test_csv_rejects_cycle_identifier_as_target(tmp_path):
    path = tmp_path / "cells.csv"
    sample().to_csv(path, index=False)
    with pytest.raises(ValueError, match="cycle or battery identifier"):
        load_generic_csv(path, target_col="cycle")
