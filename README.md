# WTX-CBO Battery RUL Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official research-code style implementation of the hybrid **Wavelet–Transformer–XGBoost (WTX)** framework with **Chaotic Billiards Optimization (CBO)** for lithium-ion battery Remaining Useful Life (RUL) prediction.

This repository corresponds to:

> **Walid Mchara, Monia Raissi.**  
> *Hybrid wavelet–transformer–XGBoost framework optimized via chaotic billiards for accurate lithium-ion battery remaining useful life prediction in electric vehicles.*  
> **Clean Energy**, 2026, 10, 119–135.  
> DOI: `10.1093/ce/zkag004`

---

## Overview

Accurate RUL prediction is essential for predictive maintenance, battery safety, and intelligent battery-management systems in electric vehicles. The WTX-CBO framework combines four complementary mechanisms:

- **Discrete Wavelet Transform (DWT)** for multiscale degradation representation
- **Transformer encoder-decoder** for long-range temporal dependency learning
- **XGBoost residual refinement** for nonlinear prediction-bias correction
- **Chaotic Billiards Optimizer (CBO)** for global hyperparameter optimization, combined with Adam for local neural-network training

The paper evaluates the framework on NASA and CALCE lithium-ion battery aging datasets.

---

## Repository structure

```text
WTX-CBO-Battery-RUL/
├── .github/
│   └── workflows/
│       └── python.yml
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── docs/
│   └── ARCHITECTURE.md
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── run_nasa.sh
│   ├── run_nasa_cbo.sh
│   └── run_quick_test.sh
├── src/
│   └── wtx_cbo/
│       ├── __init__.py
│       ├── cli.py
│       └── core.py
├── tests/
│   └── test_import.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Methodology

```text
Battery Measurements
       │
       ▼
Missing-value correction
Spline interpolation
3-point moving average
IQR outlier suppression
       │
       ▼
Pearson Correlation + Mutual Information
       │
       ▼
Min-Max Normalization
       │
       ▼
DWT: Daubechies db4, Level 4
       │
       ▼
Sliding Time Windows
       │
       ▼
Transformer Encoder-Decoder
Multi-Head Self-Attention
       │
       ├──────────────► Initial RUL prediction
       │
       ▼
XGBoost Residual Learning
       │
       ▼
Final RUL Prediction
       ▲
       │
CBO Global Hyperparameter Search
       +
Adam Local Optimization
```

---

## Paper-aligned default hyperparameters

### Transformer

| Parameter | Default |
|---|---:|
| Learning rate | `5e-4` |
| Dropout | `0.2` |
| Feed-forward dimension | `256` |
| Model dimension | `128` |
| Attention heads | `4` |
| Encoder layers | `4` |

### XGBoost

| Parameter | Default |
|---|---:|
| Subsample | `0.8` |
| Learning rate | `0.05` |
| Maximum depth | `5` |
| Number of trees | `200` |

### CBO

| Parameter | Default |
|---|---:|
| Population size | `30` |
| Maximum iterations | `100` |
| Omega | `0.9` |
| Acceleration coefficient | `1.0` |

---

## Installation

### Option 1 — requirements.txt

```bash
git clone https://github.com/walidmchara/WTX-CBO-Battery-RUL.git
cd WTX-CBO-Battery-RUL

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Windows:

```powershell
.venv\Scripts\activate
```

### Option 2 — install as a Python package

```bash
pip install -e .
```

This also creates the command:

```bash
wtx-cbo-rul
```

---

## Dataset preparation

### NASA PCoE

Place these files in `data/`:

```text
data/
├── B0005.mat
├── B0006.mat
├── B0007.mat
└── B0018.mat
```

The experiment configuration corresponding to the paper uses:

- `B0005`, `B0006`, `B0007`: training / validation
- `B0018`: exclusive held-out test battery

The loader extracts discharge-cycle information including capacity and cycle-level voltage, current and temperature statistics.

### CALCE / custom CSV

Cycle-level CSV files are also supported.

Example:

```text
cycle,voltage,current,temperature,capacity,rul
1,4.18,1.50,24.8,1.98,167
2,4.17,1.49,25.1,1.97,166
...
```

---

## Run the NASA experiment

```bash
python main.py \
  --nasa-mat data/B0005.mat data/B0006.mat data/B0007.mat data/B0018.mat \
  --test-battery B0018 \
  --lookback 32 \
  --wavelet db4 \
  --wavelet-level 4 \
  --epochs 100 \
  --output-dir outputs/nasa_b0018
```

Or:

```bash
bash scripts/run_nasa.sh
```

---

## Run with CBO optimization

```bash
python main.py \
  --nasa-mat data/B0005.mat data/B0006.mat data/B0007.mat data/B0018.mat \
  --test-battery B0018 \
  --lookback 32 \
  --wavelet db4 \
  --wavelet-level 4 \
  --cbo \
  --cbo-population 30 \
  --cbo-iterations 100 \
  --cbo-candidate-epochs 15 \
  --epochs 100 \
  --output-dir outputs/nasa_b0018_cbo
```

Or:

```bash
bash scripts/run_nasa_cbo.sh
```

Full CBO optimization is computationally expensive because every candidate hyperparameter vector requires training and validation of the hybrid model.

---

## Quick smoke test

For checking that the pipeline works before a long experiment:

```bash
bash scripts/run_quick_test.sh
```

This intentionally uses a small population, few CBO iterations and few training epochs.

---

## Generic CSV experiment

```bash
python main.py \
  --csv data/CS2_38.csv \
  --target-col rul \
  --lookback 32 \
  --wavelet db4 \
  --wavelet-level 4 \
  --epochs 100 \
  --output-dir outputs/calce_cs2_38
```

---

## Output files

A normal run creates:

```text
outputs/nasa_b0018/
├── transformer.pt
├── xgboost_residual.joblib
├── minmax_scaler.joblib
├── metadata.json
├── training_history.csv
├── test_predictions.csv
└── prediction_components.csv
```

The metadata file stores selected features, model hyperparameters, validation/test metrics, training time and measured inference latency.

---

## Evaluation metrics

The implementation reports:

- Mean Absolute Error (**MAE**)
- Mean Squared Error (**MSE**)
- Root Mean Squared Error (**RMSE**)
- Mean Absolute Percentage Error (**MAPE**)
- Coefficient of Determination (**R²**)

The paper reports average performance with MAE below `0.020`, RMSE below `0.032`, and R² above `0.98` across the evaluated battery datasets.

---

## Ablation-friendly design

The repository structure makes it straightforward to test:

- Transformer without DWT
- DWT + Transformer without XGBoost
- DWT + XGBoost without Transformer
- Full WTX without CBO
- Full WTX-CBO

These experiments correspond to the component-level analysis discussed in the paper.

---

## Reproducibility note

The paper defines the main WTX-CBO architecture, DWT settings, selected Transformer/XGBoost hyperparameters, CBO equations, and the main evaluation protocol.

However, the main article does **not** completely specify every low-level implementation choice from the supplementary algorithm. In particular:

- exact chaotic map `φ(.)`
- exact look-back length
- some CBO implementation constants
- every detail of Supplementary Algorithm S1

For transparency, this repository exposes these choices as command-line parameters. The operational CBO implementation uses a **logistic chaotic map** for `φ(.)`. This is an implementation choice, not an additional claim from the paper.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{mchara2026wtxcbo,
  title   = {Hybrid wavelet--transformer--XGBoost framework optimized via chaotic billiards for accurate lithium-ion battery remaining useful life prediction in electric vehicles},
  author  = {Mchara, Walid and Raissi, Monia},
  journal = {Clean Energy},
  volume  = {10},
  pages   = {119--135},
  year    = {2026},
  doi     = {10.1093/ce/zkag004}
}
```

GitHub also recognizes the included `CITATION.cff`.

---

## Research applications

This repository can serve as a base for research on:

- lithium-ion battery RUL prediction
- state-of-health prognostics
- battery-management systems
- multiscale time-series learning
- wavelet-enhanced Transformers
- residual boosting
- metaheuristic hyperparameter optimization
- EV predictive maintenance

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

## Authors

**Walid Mchara**  
Laboratory of Robotics, Informatics and Complex Systems (RISC)  
National Engineering School of Tunis, University of Tunis El Manar

**Monia Raissi**  
Department of Mathematics, Faculty of Science, University of Monastir

---

## Contributing

Research contributions, reproducibility improvements, bug reports and extensions are welcome through GitHub issues and pull requests.

Potential extensions include:

- uncertainty-aware RUL forecasting
- online/incremental learning
- transfer learning across battery chemistries
- Transformer alternatives such as Informer/Linformer
- SHAP and attention-based interpretability
- embedded BMS quantization and pruning
