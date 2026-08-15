#!/usr/bin/env bash
set -e

python main.py   --nasa-mat data/B0005.mat data/B0006.mat data/B0007.mat data/B0018.mat   --test-battery B0018   --lookback 32   --wavelet db4   --wavelet-level 4   --cbo   --cbo-population 30   --cbo-iterations 100   --cbo-candidate-epochs 15   --epochs 100   --output-dir outputs/nasa_b0018_cbo
