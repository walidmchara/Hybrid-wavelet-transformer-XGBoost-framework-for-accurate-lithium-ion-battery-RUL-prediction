#!/usr/bin/env bash
set -e

python main.py   --nasa-mat data/B0005.mat data/B0006.mat data/B0007.mat data/B0018.mat   --test-battery B0018   --cbo   --cbo-population 4   --cbo-iterations 3   --cbo-candidate-epochs 3   --epochs 5   --output-dir outputs/quick_test
