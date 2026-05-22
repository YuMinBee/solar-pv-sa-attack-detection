# Solar PV SA Attack Detection Experiments

This repository contains my experiment work for detecting scaling attacks in solar photovoltaic generation data.

## Scope

The dataset was provided externally.
This repository does not claim ownership of the dataset and does not include raw data.

My work in this repository focuses on:

- reproducing a CNN-LSTM paper baseline,
- implementing and evaluating residual-based attack detection experiments,
- tuning LightGBM Tweedie and CatBoost forecasters,
- comparing CatBoost and GBM-CatBoost ensemble settings,
- validating the final detector with two evaluation protocols,
- organizing final metrics and reproducible experiment scripts.

## Task

The goal is to detect scaling attacks on PV generation data.

The final detection pipeline uses:

1. A forecaster that predicts normal PV generation ratio.
2. Residual features between actual and predicted generation.
3. A zone-wise LightGBM detector that classifies attack periods.

Final selected setting:

```text
Forecaster: CatBoost
depth = 7
learning_rate = 0.03
l2_leaf_reg = 3.0
Detector: zone-wise LightGBM residual detector
```

## Dataset

Raw data is not included.

The provided dataset was organized with four PV site/capacity groups:

- `site5_5.9kw_2016_2019`
- `site5_7.0kw_2016_2019`
- `site5_327.6kw_2016_2019`
- `site5_226.8kw_2016_2019`

Attack strengths:

- SA 5%
- SA 8%
- SA 10%

The experiment used 12 generated attack CSVs:

```text
4 sites x 3 attack strengths = 12 attack files
```

## Main Results

### Fixed Comparison

| Model | SA5 F1 | SA8 F1 | SA10 F1 | Notes |
|---|---:|---:|---:|---|
| CNN-LSTM paper baseline | 0.1216 | 0.1532 | 0.1626 | 30-minute reproduction |
| tuned LightGBM Tweedie | 0.7659 | 0.8428 | 0.8579 | tuned GBM forecaster |
| CatBoost depth8 | 0.7831 | 0.8501 | 0.8682 | original CatBoost setting |
| CatBoost depth7 | 0.7864 | 0.8560 | 0.8789 | final selected model |
| GBM:CatBoost 5:5 ensemble | 0.7772 | 0.8508 | 0.8652 | lower than CatBoost-only |
| GBM:CatBoost 3:7 ensemble | 0.7709 | 0.8501 | 0.8757 | close on SA10, lower overall |

CatBoost depth7 fixed comparison:

| Attack | F1 | Recall | FPR |
|---:|---:|---:|---:|
| SA 5% | 0.7864 | 0.7374 | 0.696% |
| SA 8% | 0.8560 | 0.8513 | 0.694% |
| SA 10% | 0.8789 | 0.8941 | 0.706% |

## Validation Protocols

### 1. Within-site 6:2:2 Time Split

Each site/capacity group was split chronologically:

- first 60%: train
- next 20%: validation
- last 20%: test

This setting evaluates future attack detection on sites already seen during training.

| Attack | F1 mean | Recall mean | FPR mean |
|---:|---:|---:|---:|
| SA 5% | 0.8136 | 0.7513 | 0.482% |
| SA 8% | 0.9029 | 0.9024 | 0.483% |
| SA 10% | 0.9216 | 0.9373 | 0.486% |

### 2. Leave-One-Site-Out Validation

One entire site/capacity group was held out as the test site.
The remaining three sites were used for detector training and validation.

This setting evaluates generalization to unseen PV sites.

| Attack | F1 mean | Recall mean | FPR mean |
|---:|---:|---:|---:|
| SA 5% | 0.7493 | 0.7915 | 1.692% |
| SA 8% | 0.8046 | 0.8896 | 1.695% |
| SA 10% | 0.8167 | 0.9125 | 1.697% |

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── results/
│   ├── paper_cnn_lstm_30min_metrics_summary.csv
│   ├── tuned_lightgbm_tweedie_detector_metrics.csv
│   ├── catboost_depth8_detector_metrics.csv
│   ├── catboost_depth7_detector_metrics.csv
│   ├── catboost_depth_tuning_summary.csv
│   ├── catboost_lr_tuning_cpu_summary.csv
│   ├── depth7_lgbm_ensemble_detector_summary.csv
│   ├── depth7_lgbm_ensemble_vs_depth7_comparison.csv
│   ├── within_kw_622_summary_by_attack_ratio.csv
│   ├── within_kw_622_metrics.csv
│   ├── within_kw_622_split_counts.csv
│   ├── site_kfold_summary_by_attack_ratio.csv
│   └── site_kfold_metrics.csv
└── scripts/
    ├── paper_cnn_lstm_baseline.py
    ├── test_ver7_catboost_depth7.py
    ├── run_catboost_depth_only_tuning.py
    ├── run_catboost_lr_only_tuning.py
    ├── run_depth7_lgbm_ensemble_weights.py
    ├── run_catboost_depth7_within_kw_622.py
    └── run_catboost_depth7_site_kfold.py
```

## Reproduction Notes

The scripts assume the original project workspace contains:

```text
dataset_clean/
generated_attack_data/
model_output_catboost_depth_only/
```

These directories are intentionally excluded from this repository.

Install dependencies:

```bash
pip install -r requirements.txt
```

Example commands:

```bash
python scripts/run_catboost_depth_only_tuning.py
python scripts/run_catboost_depth7_within_kw_622.py
python scripts/run_catboost_depth7_site_kfold.py
```

Path adjustments may be needed outside the original experiment workspace.

## Conclusion

The provided dataset was used to reproduce a paper baseline and evaluate multiple residual-based attack detection settings.
CatBoost depth7 with a zone-wise residual detector achieved the best overall balance.
LightGBM Tweedie tuning improved the GBM baseline, but CatBoost depth7 achieved higher F1 and lower FPR.
GBM-CatBoost weighted ensembles did not improve the final result.

The within-site 6:2:2 split is the main operational performance result.
The leave-one-site-out result is a stricter generalization check.
