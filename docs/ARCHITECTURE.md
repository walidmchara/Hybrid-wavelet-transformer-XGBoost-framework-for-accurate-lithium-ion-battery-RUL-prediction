# Architecture

The implementation follows the paper's main processing chain:

1. Battery cycle measurements
2. Cleaning and smoothing
3. Pearson correlation + mutual-information feature screening
4. Min-Max scaling
5. DWT multiscale decomposition using `db4`, level 4
6. Sliding-window sequence construction
7. Encoder-decoder Transformer
8. XGBoost residual correction
9. Optional CBO hyperparameter search
10. Final RUL prediction and evaluation

The Transformer learns long-range temporal dependencies while XGBoost models residual nonlinear error. CBO performs global hyperparameter search and Adam performs local neural-network weight optimization.

## Important reproducibility note

The main paper does not fully specify every implementation-level constant, including the exact chaotic map used for `phi(.)`, the exact look-back length, or every value in Supplementary Algorithm S1. The repository therefore exposes those values as parameters. The default operational implementation uses a logistic chaotic map for the CBO perturbation.
