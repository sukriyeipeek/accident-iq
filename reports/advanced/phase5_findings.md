# Phase 5 — Advanced ML

Fixed Random Forest and LightGBM experiments use the same raw-input, fold-fitted preprocessing. The 2019 temporal folds alone select the advanced model; 2020 is held out for validation. No sampling or parameter search was used. Class weighting changes probability calibration and the precision/recall trade-off. These models estimate severity conditional on a recorded injury collision, not collision occurrence. 2021 was not scored.

| Model | CV macro F1 | 2020 macro F1 | Accuracy | Fatal recall | Serious recall | Log-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_forest_balanced | 0.3701 | 0.3667 | 0.6000 | 0.4206 | 0.2713 | 0.8911 |
| lightgbm | 0.2945 | 0.2929 | 0.7835 | 0.0000 | 0.0001 | 0.5601 |
| lightgbm_balanced | 0.3509 | 0.3451 | 0.5287 | 0.4881 | 0.3240 | 0.9386 |

Selected by 2019 CV macro F1: `random_forest_balanced`. CV standard deviation is descriptive across three overlapping expanding windows, not a confidence interval. Full class metrics are in `metrics.json`; parameters, timing and checks are in `summary.json`. The selected artifact is `models/advanced_v1/selected.joblib`.

Learning notes: Random Forest averages randomized trees; LightGBM builds trees sequentially to reduce residual error. Macro F1 weights severity classes equally. Class weights affect the training objective and can raise minority recall while increasing false positives.
