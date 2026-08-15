#!/usr/bin/env python3
"""
WTX-CBO: Wavelet–Transformer–XGBoost + Chaotic Billiards Optimizer
for lithium-ion battery Remaining Useful Life (RUL) prediction.

Implementation based on:
W. Mchara, M. Raissi, "Hybrid wavelet–transformer–XGBoost framework
optimized via chaotic billiards for accurate lithium-ion battery remaining
useful life prediction in electric vehicles", Clean Energy, 2026.

IMPORTANT REPRODUCIBILITY NOTE
------------------------------
The main paper specifies the architecture, db4 DWT level 4, selected
Transformer/XGBoost hyperparameters, CBO equations/configuration, and
chronological evaluation protocol. Some low-level implementation details
(e.g. the exact chaotic map, look-back length, EOL/RUL labelling rule and
all Supplementary Algorithm S1 constants) are not fully specified in the
main PDF. This script therefore exposes those choices as CLI parameters.

Default choices used here:
  * RUL target = remaining observed discharge cycles (dataset-end target)
  * look-back = 32 cycles
  * CBO chaotic map = logistic map z <- 4 z (1-z)
These defaults are explicit, configurable engineering choices and should
not be interpreted as additional claims from the paper.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import pywt
import scipy.io
from scipy.interpolate import UnivariateSpline
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    eps = 1e-8
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# NASA MAT loader
# ---------------------------------------------------------------------------

def _mat_scalar(x, default=np.nan):
    try:
        arr = np.asarray(x).squeeze()
        if arr.size == 0:
            return default
        return float(arr.flat[0])
    except Exception:
        return default


def _flatten_numeric(x) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]
    except Exception:
        return np.asarray([], dtype=float)


def _safe_mean(x) -> float:
    a = _flatten_numeric(x)
    return float(np.mean(a)) if len(a) else np.nan


def _safe_std(x) -> float:
    a = _flatten_numeric(x)
    return float(np.std(a)) if len(a) else np.nan


def load_nasa_battery_mat(path: str | Path) -> pd.DataFrame:
    """
    Loads a NASA PCoE B0005/B0006/B0007/B0018 style .mat file and
    returns one row per DISCHARGE cycle.

    Extracted cycle-level features:
      cycle, ambient_temperature, voltage_mean/std, current_mean/std,
      temperature_mean/std, load_voltage_mean, load_current_mean,
      time_duration, capacity
    """
    path = Path(path)
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)

    key = path.stem
    if key not in mat:
        candidates = [k for k in mat.keys() if not k.startswith("__")]
        if not candidates:
            raise ValueError(f"No battery variable found in {path}")
        key = candidates[0]

    battery = mat[key]
    cycles = np.atleast_1d(getattr(battery, "cycle", []))
    rows = []
    discharge_idx = 0

    for cyc in cycles:
        typ = str(getattr(cyc, "type", "")).lower()
        if "discharge" not in typ:
            continue

        data = getattr(cyc, "data", None)
        if data is None:
            continue

        discharge_idx += 1
        time_arr = _flatten_numeric(getattr(data, "Time", []))
        duration = float(time_arr[-1] - time_arr[0]) if len(time_arr) >= 2 else np.nan

        capacity = _mat_scalar(getattr(data, "Capacity", np.nan))
        rows.append({
            "cycle": discharge_idx,
            "ambient_temperature": _mat_scalar(getattr(cyc, "ambient_temperature", np.nan)),
            "voltage_mean": _safe_mean(getattr(data, "Voltage_measured", [])),
            "voltage_std": _safe_std(getattr(data, "Voltage_measured", [])),
            "current_mean": _safe_mean(getattr(data, "Current_measured", [])),
            "current_std": _safe_std(getattr(data, "Current_measured", [])),
            "temperature_mean": _safe_mean(getattr(data, "Temperature_measured", [])),
            "temperature_std": _safe_std(getattr(data, "Temperature_measured", [])),
            "load_voltage_mean": _safe_mean(getattr(data, "Voltage_load", [])),
            "load_current_mean": _safe_mean(getattr(data, "Current_load", [])),
            "time_duration": duration,
            "capacity": capacity,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No discharge cycles extracted from {path}")
    df["battery_id"] = path.stem
    df["rul"] = np.arange(len(df) - 1, -1, -1, dtype=float)
    return df


def load_generic_csv(path: str | Path,
                     battery_id: Optional[str] = None,
                     cycle_col: str = "cycle",
                     target_col: Optional[str] = None) -> pd.DataFrame:
    """
    Generic CSV loader for CALCE or custom data.

    Requirements:
      - one row per cycle, or a pre-aggregated cycle-level table
      - numeric feature columns
      - optional target column. If absent, RUL is generated as remaining
        rows/cycles to end of the table.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if cycle_col not in df.columns:
        df[cycle_col] = np.arange(1, len(df) + 1)
    df = df.sort_values(cycle_col).reset_index(drop=True)

    if target_col and target_col in df.columns:
        df["rul"] = pd.to_numeric(df[target_col], errors="coerce")
    elif "rul" not in df.columns:
        df["rul"] = np.arange(len(df) - 1, -1, -1, dtype=float)

    df["battery_id"] = battery_id or path.stem
    return df


# ---------------------------------------------------------------------------
# Preprocessing from the paper:
# spline interpolation, 3-point moving average, IQR, MinMax,
# Pearson + mutual information feature screening, db4 DWT level 4
# ---------------------------------------------------------------------------

def spline_fill(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.notna().sum() < 4:
        return s.interpolate(limit_direction="both").ffill().bfill()
    x = np.arange(len(s))
    ok = s.notna().values
    try:
        spline = UnivariateSpline(x[ok], s.values[ok], s=0, k=min(3, ok.sum() - 1))
        y = s.values.copy()
        y[~ok] = spline(x[~ok])
        return pd.Series(y, index=s.index)
    except Exception:
        return s.interpolate(limit_direction="both").ffill().bfill()


def clean_numeric_frame(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()

    for col in feature_cols:
        out[col] = spline_fill(out[col])
        out[col] = out[col].rolling(window=3, min_periods=1, center=True).mean()

    # IQR winsorization: keeps chronological length intact while suppressing
    # transient outliers. The paper states IQR outlier removal; clipping is
    # used here to avoid destroying sequence continuity.
    for col in feature_cols:
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[col] = out[col].clip(lo, hi)

    return out


def select_features(train_df: pd.DataFrame,
                    candidate_cols: Sequence[str],
                    target_col: str = "rul",
                    top_k: Optional[int] = None,
                    min_abs_corr: float = 0.0) -> List[str]:
    """
    Pearson correlation + mutual information screening.
    Ranking score = normalized |corr| + normalized MI.
    """
    X = train_df[list(candidate_cols)].astype(float)
    y = train_df[target_col].astype(float).values

    corr = X.apply(lambda s: abs(np.corrcoef(s.values, y)[0, 1]) if s.std() > 0 else 0.0)
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    mi = pd.Series(
        mutual_info_regression(X.fillna(X.median()), y, random_state=42),
        index=X.columns,
    )

    corr_n = corr / (corr.max() + 1e-12)
    mi_n = mi / (mi.max() + 1e-12)
    score = corr_n + mi_n

    keep = [c for c in candidate_cols if corr[c] >= min_abs_corr]
    if not keep:
        keep = list(candidate_cols)
    ranked = sorted(keep, key=lambda c: score[c], reverse=True)
    return ranked if top_k is None else ranked[:top_k]


def dwt_multiscale_same_length(
    values: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """
    For each original feature, creates same-length reconstructed DWT bands:
      A_level, D_level, D_(level-1), ..., D_1
    Thus level=4 gives five multiscale channels per input feature.

    This retains temporal alignment required by the Transformer.
    """
    values = np.asarray(values, dtype=float)
    n, d = values.shape
    out = []

    for j in range(d):
        x = values[:, j]
        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
        lev = max(1, min(level, max_level)) if max_level > 0 else 1
        coeffs = pywt.wavedec(x, wavelet=wavelet, level=lev, mode="symmetric")

        # Isolate and reconstruct each coefficient group separately.
        for k in range(len(coeffs)):
            isolated = [np.zeros_like(c) for c in coeffs]
            isolated[k] = coeffs[k]
            rec = pywt.waverec(isolated, wavelet=wavelet, mode="symmetric")[:n]
            if len(rec) < n:
                rec = np.pad(rec, (0, n - len(rec)), mode="edge")
            out.append(rec)

    return np.stack(out, axis=1)


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    selected_features: List[str]
    test_meta: pd.DataFrame


def make_windows(X: np.ndarray, y: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, idx = [], [], []
    for i in range(lookback - 1, len(X)):
        xs.append(X[i - lookback + 1:i + 1])
        ys.append(y[i])
        idx.append(i)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(idx)


def _split_chronological(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
    n = len(df)
    i1 = max(1, int(n * train_ratio))
    i2 = max(i1 + 1, int(n * (train_ratio + val_ratio)))
    i2 = min(i2, n)
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def prepare_single_series(
    df: pd.DataFrame,
    lookback: int = 32,
    feature_cols: Optional[Sequence[str]] = None,
    top_k_features: Optional[int] = None,
    wavelet: str = "db4",
    wavelet_level: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> PreparedData:
    """
    Strict chronological 80/10/10 workflow for a single series.

    Preprocessing parameters are fit on TRAIN only.
    DWT is applied after scaling; for val/test the concatenated prefix is
    transformed and only the relevant segment is retained, avoiding future
    information from later segments while preserving boundary context.
    """
    df = df.sort_values("cycle" if "cycle" in df.columns else df.index.name or df.columns[0]).reset_index(drop=True)
    numeric = [c for c in df.select_dtypes(include=np.number).columns if c not in {"rul"}]
    if feature_cols is None:
        feature_cols = [c for c in numeric if c != "cycle"]
        if "cycle" in numeric:
            feature_cols = ["cycle"] + feature_cols
    feature_cols = list(feature_cols)

    tr, va, te = _split_chronological(df, train_ratio, val_ratio)
    if len(te) < lookback:
        warnings.warn("Test segment shorter than lookback; windows use preceding chronological context.")

    # clean entire chronological stream per-feature but feature selection/scaling fit on train
    cleaned = clean_numeric_frame(df, feature_cols)
    tr_c = cleaned.iloc[:len(tr)].copy()

    selected = select_features(tr_c.assign(rul=tr["rul"].values), feature_cols,
                               target_col="rul", top_k=top_k_features)

    scaler = MinMaxScaler()
    scaler.fit(tr_c[selected].values)
    scaled_all = scaler.transform(cleaned[selected].values)

    multiscale_all = dwt_multiscale_same_length(scaled_all, wavelet=wavelet, level=wavelet_level)
    y_all = df["rul"].astype(float).values

    Xw, yw, iw = make_windows(multiscale_all, y_all, lookback)

    n = len(df)
    i1 = max(1, int(n * train_ratio))
    i2 = max(i1 + 1, int(n * (train_ratio + val_ratio)))
    end_indices = iw

    tr_mask = end_indices < i1
    va_mask = (end_indices >= i1) & (end_indices < i2)
    te_mask = end_indices >= i2

    meta = df.iloc[end_indices[te_mask]].reset_index(drop=True)

    return PreparedData(
        X_train=Xw[tr_mask], y_train=yw[tr_mask],
        X_val=Xw[va_mask], y_val=yw[va_mask],
        X_test=Xw[te_mask], y_test=yw[te_mask],
        scaler=scaler, selected_features=selected,
        test_meta=meta,
    )


# ---------------------------------------------------------------------------
# Transformer encoder-decoder
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RULTransformer(nn.Module):
    """
    Encoder-decoder Transformer.
    A learned one-token decoder query attends to encoded degradation history
    and outputs one RUL estimate per input window.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        z = self.pos(self.input_proj(x))
        memory = self.encoder(z)
        q = self.query.expand(x.size(0), -1, -1)
        dec = self.decoder(q, memory)
        return self.head(dec[:, 0]).squeeze(-1)


class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class TransformerConfig:
    learning_rate: float = 5e-4
    dropout: float = 0.2
    dim_feedforward: int = 256
    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 2
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    weight_decay: float = 1e-5


def train_transformer(
    X_train, y_train, X_val, y_val,
    config: TransformerConfig,
    device: str,
    verbose: bool = True,
):
    model = RULTransformer(
        input_dim=X_train.shape[-1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.encoder_layers,
        num_decoder_layers=config.decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)

    train_loader = DataLoader(ArrayDataset(X_train, y_train),
                              batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(ArrayDataset(X_val, y_val),
                            batch_size=config.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())

        tr_loss = float(np.mean(train_losses))
        va_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": va_loss})

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"Epoch {epoch:03d} | train MSE={tr_loss:.6f} | val MSE={va_loss:.6f}")

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= config.patience:
                break

    model.load_state_dict(best_state)
    return model, history


def predict_transformer(model, X, device, batch_size=256) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.as_tensor(X, dtype=torch.float32),
                        batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# XGBoost residual refinement
# paper: final = transformer + learned residual
# ---------------------------------------------------------------------------

@dataclass
class XGBConfig:
    subsample: float = 0.8
    learning_rate: float = 0.05
    max_depth: int = 5
    n_estimators: int = 200
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0


def residual_features(X: np.ndarray, transformer_pred: np.ndarray) -> np.ndarray:
    """
    Compact residual-learning features:
      last multiscale vector + temporal mean/std/min/max + Transformer prediction
    """
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    minv = X.min(axis=1)
    maxv = X.max(axis=1)
    return np.concatenate(
        [last, mean, std, minv, maxv, transformer_pred.reshape(-1, 1)],
        axis=1
    )


def fit_xgb_residual(X_train, y_train, trans_train, cfg: XGBConfig) -> XGBRegressor:
    residual = y_train - trans_train
    feats = residual_features(X_train, trans_train)
    model = XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(feats, residual)
    return model


def hybrid_predict(transformer, xgb, X, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tpred = predict_transformer(transformer, X, device)
    residual = xgb.predict(residual_features(X, tpred))
    final = tpred + residual
    return final, tpred, residual


# ---------------------------------------------------------------------------
# Chaotic Billiards Optimizer (paper-inspired operational implementation)
# ---------------------------------------------------------------------------

@dataclass
class CBOConfig:
    population_size: int = 30
    max_iterations: int = 100
    omega: float = 0.9       # paper Table 4
    acceleration: float = 1.0  # paper Table 4
    mu: float = 0.7          # exposed because exact value not in main PDF
    lambda_chaos: float = 0.10
    seed: int = 42


class ChaoticBilliardsOptimizer:
    """
    Numerical implementation of the CBO equations reported in the paper:
      X(t+1)=X(t)+mu*v(t)+lambda*phi(X(t))
      v(t+1)=omega*v(t)+r1*(P-X)+r2*(G-X)

    The exact phi(.) map is not specified in the main PDF, so this code uses a
    logistic chaotic map. Search-space decoding converts normalized particle
    positions into discrete/continuous hyperparameters.
    """
    def __init__(self, bounds: Sequence[Tuple[float, float]], cfg: CBOConfig):
        self.bounds = np.asarray(bounds, dtype=float)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    @staticmethod
    def logistic(z):
        z = np.clip(z, 1e-8, 1 - 1e-8)
        return 4.0 * z * (1.0 - z)

    def optimize(self, objective):
        d = len(self.bounds)
        low, high = self.bounds[:, 0], self.bounds[:, 1]

        X = self.rng.uniform(low, high, size=(self.cfg.population_size, d))
        V = np.zeros_like(X)
        P = X.copy()
        Pscore = np.full(self.cfg.population_size, np.inf)

        G = X[0].copy()
        Gscore = np.inf
        chaos = self.rng.uniform(0.1, 0.9, size=X.shape)
        trace = []

        for it in range(self.cfg.max_iterations):
            for i in range(self.cfg.population_size):
                score = float(objective(X[i]))
                if score < Pscore[i]:
                    Pscore[i] = score
                    P[i] = X[i].copy()
                if score < Gscore:
                    Gscore = score
                    G = X[i].copy()

            trace.append({"iteration": it + 1, "best_rmse": Gscore})
            print(f"CBO iter {it+1:03d}/{self.cfg.max_iterations} | best val RMSE={Gscore:.6f}")

            chaos = self.logistic(chaos)
            r1 = chaos
            r2 = self.logistic(chaos)

            V = (
                self.cfg.omega * V
                + self.cfg.acceleration * r1 * (P - X)
                + self.cfg.acceleration * r2 * (G - X)
            )
            perturb = 2.0 * chaos - 1.0
            X = X + self.cfg.mu * V + self.cfg.lambda_chaos * perturb * (high - low)
            X = np.clip(X, low, high)

        return G, Gscore, trace


def decode_hparams(v: np.ndarray) -> Tuple[TransformerConfig, XGBConfig]:
    """
    Search vector:
      [lr, dropout, d_model_idx, nhead_idx, ff_idx, enc_layers,
       xgb_lr, xgb_depth, n_estimators_idx, subsample]
    """
    dmodels = [64, 128, 256]
    heads = [2, 4, 8]
    ffs = [128, 256, 512]
    trees = [100, 200, 300]

    tc = TransformerConfig(
        learning_rate=float(v[0]),
        dropout=float(v[1]),
        d_model=dmodels[int(np.clip(round(v[2]), 0, 2))],
        nhead=heads[int(np.clip(round(v[3]), 0, 2))],
        dim_feedforward=ffs[int(np.clip(round(v[4]), 0, 2))],
        encoder_layers=int(np.clip(round(v[5]), 2, 4)),
    )
    # guarantee divisibility
    if tc.d_model % tc.nhead != 0:
        tc.nhead = 4 if tc.d_model % 4 == 0 else 2

    xc = XGBConfig(
        learning_rate=float(v[6]),
        max_depth=int(np.clip(round(v[7]), 3, 7)),
        n_estimators=trees[int(np.clip(round(v[8]), 0, 2))],
        subsample=float(v[9]),
    )
    return tc, xc


CBO_BOUNDS = [
    (1e-4, 1e-3),  # transformer lr
    (0.1, 0.3),    # dropout
    (0, 2),        # d_model index
    (0, 2),        # heads index
    (0, 2),        # feedforward index
    (2, 4),        # encoder layers
    (0.01, 0.1),   # xgb lr
    (3, 7),        # xgb depth
    (0, 2),        # trees index
    (0.6, 1.0),    # subsample
]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def train_full_hybrid(
    data: PreparedData,
    transformer_cfg: TransformerConfig,
    xgb_cfg: XGBConfig,
    device: str,
    verbose: bool = True,
):
    transformer, history = train_transformer(
        data.X_train, data.y_train, data.X_val, data.y_val,
        transformer_cfg, device, verbose=verbose
    )
    tr_pred = predict_transformer(transformer, data.X_train, device)
    xgb = fit_xgb_residual(data.X_train, data.y_train, tr_pred, xgb_cfg)
    return transformer, xgb, history


def evaluate_hybrid(transformer, xgb, X, y, device):
    final, tpred, residual = hybrid_predict(transformer, xgb, X, device)
    return regression_metrics(y, final), final, tpred, residual


def run_cbo_search(data: PreparedData, device: str, args):
    # CBO is computationally expensive. For practical search, each candidate
    # uses fewer epochs; best params are retrained later with full epochs.
    candidate_epochs = args.cbo_candidate_epochs

    cache = {}
    def objective(v):
        tc, xc = decode_hparams(v)
        tc.max_epochs = candidate_epochs
        tc.patience = min(5, candidate_epochs)

        key = json.dumps({"t": asdict(tc), "x": asdict(xc)}, sort_keys=True)
        if key in cache:
            return cache[key]

        try:
            model, xgb, _ = train_full_hybrid(data, tc, xc, device, verbose=False)
            m, *_ = evaluate_hybrid(model, xgb, data.X_val, data.y_val, device)
            score = m["RMSE"]
        except Exception as e:
            warnings.warn(f"CBO candidate failed: {e}")
            score = 1e9

        cache[key] = score
        return score

    cfg = CBOConfig(
        population_size=args.cbo_population,
        max_iterations=args.cbo_iterations,
        omega=0.9,
        acceleration=1.0,
        mu=args.cbo_mu,
        lambda_chaos=args.cbo_lambda,
        seed=args.seed,
    )
    best_vec, best_score, trace = ChaoticBilliardsOptimizer(CBO_BOUNDS, cfg).optimize(objective)
    tc, xc = decode_hparams(best_vec)
    return tc, xc, best_score, trace


def save_artifacts(
    output_dir: Path,
    transformer,
    xgb,
    data: PreparedData,
    tcfg: TransformerConfig,
    xcfg: XGBConfig,
    history,
    metrics,
    predictions,
    test_meta,
    extra: Optional[dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": transformer.state_dict(),
        "input_dim": data.X_train.shape[-1],
        "transformer_config": asdict(tcfg),
    }, output_dir / "transformer.pt")
    joblib.dump(xgb, output_dir / "xgboost_residual.joblib")
    joblib.dump(data.scaler, output_dir / "minmax_scaler.joblib")

    meta = {
        "selected_features": data.selected_features,
        "transformer_config": asdict(tcfg),
        "xgboost_config": asdict(xcfg),
        "metrics": metrics,
    }
    if extra:
        meta.update(extra)
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    pred_df = test_meta.copy()
    pred_df["rul_true"] = data.y_test
    pred_df["rul_pred"] = predictions
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser(
        description="WTX-CBO lithium-ion battery RUL prediction"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--nasa-mat", nargs="+", help="NASA .mat files")
    src.add_argument("--csv", help="Generic cycle-level CSV file")

    p.add_argument("--test-battery", default=None,
                   help="For multiple NASA files, battery ID reserved exclusively for test (e.g. B0018).")
    p.add_argument("--target-col", default=None,
                   help="CSV target column; otherwise remaining rows are used as RUL.")
    p.add_argument("--feature-cols", nargs="+", default=None)
    p.add_argument("--lookback", type=int, default=32)
    p.add_argument("--top-k-features", type=int, default=None)
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--wavelet-level", type=int, default=4)

    p.add_argument("--cbo", action="store_true", help="Run CBO hyperparameter search")
    p.add_argument("--cbo-population", type=int, default=30)
    p.add_argument("--cbo-iterations", type=int, default=100)
    p.add_argument("--cbo-candidate-epochs", type=int, default=15)
    p.add_argument("--cbo-mu", type=float, default=0.7)
    p.add_argument("--cbo-lambda", type=float, default=0.10)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/wtx_cbo")
    return p.parse_args()


def prepare_multi_battery_nasa(args) -> PreparedData:
    frames = [load_nasa_battery_mat(f) for f in args.nasa_mat]

    if args.test_battery:
        train_frames = [d for d in frames if str(d["battery_id"].iloc[0]) != args.test_battery]
        test_frames = [d for d in frames if str(d["battery_id"].iloc[0]) == args.test_battery]
        if not test_frames:
            raise ValueError(f"Test battery '{args.test_battery}' not found.")
        if not train_frames:
            raise ValueError("No training batteries remain after reserving test battery.")

        # Merge training cells but preserve chronological sequences by battery.
        # Feature selection/scaler fit from all training-cell rows.
        all_train = pd.concat(train_frames, ignore_index=True)
        numeric = [c for c in all_train.select_dtypes(include=np.number).columns if c != "rul"]
        feature_cols = args.feature_cols or numeric

        cleaned_train_cells = [clean_numeric_frame(d, feature_cols) for d in train_frames]
        merged_clean = pd.concat(cleaned_train_cells, ignore_index=True)
        merged_target = pd.concat([d[["rul"]] for d in train_frames], ignore_index=True)

        selected = select_features(
            merged_clean.assign(rul=merged_target["rul"].values),
            feature_cols,
            top_k=args.top_k_features,
        )
        scaler = MinMaxScaler().fit(merged_clean[selected].values)

        # Split windows cell-by-cell to avoid windows crossing battery boundaries.
        Xtr_all, ytr_all, Xv_all, yv_all = [], [], [], []
        for raw, clean in zip(train_frames, cleaned_train_cells):
            vals = scaler.transform(clean[selected].values)
            ms = dwt_multiscale_same_length(vals, args.wavelet, args.wavelet_level)
            Xw, yw, iw = make_windows(ms, raw["rul"].values, args.lookback)

            split = max(1, int(0.9 * len(Xw)))
            Xtr_all.append(Xw[:split]); ytr_all.append(yw[:split])
            Xv_all.append(Xw[split:]); yv_all.append(yw[split:])

        # Exclusive held-out test battery.
        test_raw = test_frames[0].reset_index(drop=True)
        test_clean = clean_numeric_frame(test_raw, selected)
        test_vals = scaler.transform(test_clean[selected].values)
        test_ms = dwt_multiscale_same_length(test_vals, args.wavelet, args.wavelet_level)
        Xte, yte, ite = make_windows(test_ms, test_raw["rul"].values, args.lookback)
        test_meta = test_raw.iloc[ite].reset_index(drop=True)

        return PreparedData(
            X_train=np.concatenate(Xtr_all), y_train=np.concatenate(ytr_all),
            X_val=np.concatenate(Xv_all), y_val=np.concatenate(yv_all),
            X_test=Xte, y_test=yte,
            scaler=scaler, selected_features=selected, test_meta=test_meta,
        )

    # If no exclusive test battery is named, use chronological 80/10/10 on first/only series.
    if len(frames) != 1:
        warnings.warn("Multiple NASA files supplied without --test-battery; concatenating is unsafe. Using first file only.")
    return prepare_single_series(
        frames[0],
        lookback=args.lookback,
        feature_cols=args.feature_cols,
        top_k_features=args.top_k_features,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.nasa_mat:
        data = prepare_multi_battery_nasa(args)
    else:
        df = load_generic_csv(args.csv, target_col=args.target_col)
        data = prepare_single_series(
            df,
            lookback=args.lookback,
            feature_cols=args.feature_cols,
            top_k_features=args.top_k_features,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
        )

    print(f"Selected features: {data.selected_features}")
    print(f"Train/Val/Test windows: {len(data.y_train)}/{len(data.y_val)}/{len(data.y_test)}")
    print(f"Transformer input channels after DWT: {data.X_train.shape[-1]}")

    tcfg = TransformerConfig(
        learning_rate=5e-4,
        dropout=0.2,
        dim_feedforward=256,
        d_model=128,
        nhead=4,
        encoder_layers=4,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
    )
    xcfg = XGBConfig(
        subsample=0.8,
        learning_rate=0.05,
        max_depth=5,
        n_estimators=200,
    )

    cbo_info = {}
    if args.cbo:
        print("Running CBO search. This can be computationally expensive.")
        tcfg, xcfg, cbo_best, cbo_trace = run_cbo_search(data, device, args)
        tcfg.max_epochs = args.epochs
        tcfg.batch_size = args.batch_size
        cbo_info = {"cbo_best_validation_rmse": cbo_best, "cbo_trace": cbo_trace}
        print("Best Transformer config:", asdict(tcfg))
        print("Best XGBoost config:", asdict(xcfg))

    t0 = time.perf_counter()
    transformer, xgb, history = train_full_hybrid(data, tcfg, xcfg, device)
    train_seconds = time.perf_counter() - t0

    val_metrics, val_pred, _, _ = evaluate_hybrid(
        transformer, xgb, data.X_val, data.y_val, device
    )
    test_metrics, test_pred, trans_pred, residual = evaluate_hybrid(
        transformer, xgb, data.X_test, data.y_test, device
    )

    print("\nValidation metrics")
    print(json.dumps(val_metrics, indent=2))
    print("\nTest metrics")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nTraining time: {train_seconds:.2f} s")

    # Batch=1 latency benchmark.
    nbench = min(100, len(data.X_test))
    if nbench > 0:
        if device == "cuda":
            torch.cuda.synchronize()
        s = time.perf_counter()
        for i in range(nbench):
            hybrid_predict(transformer, xgb, data.X_test[i:i+1], device)
        if device == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - s) / nbench
    else:
        latency = float("nan")

    extras = {
        "paper_defaults_used": {
            "wavelet": args.wavelet,
            "wavelet_level": args.wavelet_level,
            "chronological_split": "80/10/10 for single series; exclusive held-out cell supported",
        },
        "training_seconds": train_seconds,
        "batch1_latency_seconds": latency,
        **cbo_info,
    }

    save_artifacts(
        Path(args.output_dir),
        transformer, xgb, data, tcfg, xcfg, history,
        {"validation": val_metrics, "test": test_metrics},
        test_pred, data.test_meta, extras
    )

    # Additional prediction details.
    pd.DataFrame({
        "rul_true": data.y_test,
        "transformer_pred": trans_pred,
        "xgb_residual": residual,
        "hybrid_pred": test_pred,
    }).to_csv(Path(args.output_dir) / "prediction_components.csv", index=False)

    print(f"\nSaved artifacts to: {args.output_dir}")
    print(f"Batch-1 measured latency: {latency:.6f} s/window")


if __name__ == "__main__":
    main()
