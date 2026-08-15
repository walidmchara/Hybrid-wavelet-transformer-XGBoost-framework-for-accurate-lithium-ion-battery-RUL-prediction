#!/usr/bin/env bash
set -e

python main.py   --nasa-mat data/B0005.mat data/B0006.mat data/B0007.mat data/B0018.mat   --test-battery B0018   --lookback 32   --wavelet db4   --wavelet-level 4   --epochs 100   --output-dir outputs/nasa_b0018
