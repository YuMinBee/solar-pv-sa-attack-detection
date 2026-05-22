#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost depth7 forecaster + detector leave-one-site-out k-fold experiment.

경로 설명
---------
- 이 스크립트는 현재 폴더 기준으로 실행한다.
- 입력 공격 CSV:
  ./generated_attack_data/site5_*_attack_sa_{5,8,10}pct.csv
- 재사용 코드:
  ./test_ver7_catboost_depth7.py
- 출력 폴더:
  ./detector_model_fadre_v3_catboost_depth7_site_kfold
- 캐시 폴더:
  ./feature_cache_fadre_v3_catboost_depth7_site_kfold

실험 설계
---------
- 4개 site 중 1개 site를 통째로 test로 제외한다.
- 나머지 3개 site의 SA 5%, SA 10%만 detector 학습/검증에 사용한다.
- 제외한 test site는 SA 5%, SA 8%, SA 10%로 평가한다.
- SA 8%는 학습에 넣지 않아, 중간 공격 강도 일반화 성능을 확인한다.
"""

import json
import os
from pathlib import Path

import pandas as pd

import test_ver7_catboost_depth7 as base


SITES = [
    "site5_5.9kw_2016_2019",
    "site5_7.0kw_2016_2019",
    "site5_327.6kw_2016_2019",
    "site5_226.8kw_2016_2019",
]

TRAIN_ATTACK_PCTS = [5, 10]
TEST_ATTACK_PCTS = [5, 8, 10]

OUT_DIR = "./detector_model_fadre_v3_catboost_depth7_site_kfold"
CACHE_DIR = "./feature_cache_fadre_v3_catboost_depth7_site_kfold"


def attack_csv(site, pct):
    return f"./generated_attack_data/{site}_attack_sa_{pct}pct.csv"


def metric_row(fold, test_site, dataset, metric):
    row = {
        "fold": fold,
        "test_site": test_site,
        "dataset": dataset,
    }
    row.update(metric)
    return row


def run_fold(fold, test_site):
    train_sites = [s for s in SITES if s != test_site]
    train_csvs = [attack_csv(site, pct) for site in train_sites for pct in TRAIN_ATTACK_PCTS]
    test_csvs = [attack_csv(test_site, pct) for pct in TEST_ATTACK_PCTS]

    print("\n" + "=" * 90)
    print(f"[Fold {fold}] test_site={test_site}")
    print("=" * 90)
    print("train_csvs:")
    for p in train_csvs:
        print(f"  - {p}")
    print("test_csvs:")
    for p in test_csvs:
        print(f"  - {p}")

    base.TRAIN_ATTACK_CSVS = train_csvs
    base.EVAL_CSVS_SA = test_csvs
    base.OUT_DIR = OUT_DIR
    base.CACHE_DIR = CACHE_DIR

    fc = base.ForecasterWrapper().load()
    base.log("Residual centering/calibration disabled: using raw forecaster predictions only")

    df_all = base.build_train_features(fc)
    unique_dates = sorted(df_all["date"].unique())
    split_idx = int(len(unique_dates) * base.TRAIN_RATIO)
    train_dates = set(unique_dates[:split_idx])
    val_dates = set(unique_dates[split_idx:])

    dtr = df_all[df_all["date"].isin(train_dates)].reset_index(drop=True)
    dva = df_all[df_all["date"].isin(val_dates)].reset_index(drop=True)
    base.log(
        f"fold={fold} train={len(dtr):,}(pos={int(dtr['is_attack'].sum())}) "
        f"val={len(dva):,}(pos={int(dva['is_attack'].sum())})"
    )

    zfc, drop = base.select_zone_features(dtr)
    dtr = dtr.drop(columns=drop, errors="ignore")
    dva = dva.drop(columns=drop, errors="ignore")
    base.log(
        f"fold={fold} dropped {len(drop)} common features | "
        f"high={len(zfc['high'])} mid={len(zfc['mid'])}"
    )

    models, val_prob = base.train_zone_models(dtr, dva, zfc)
    val_dec = base.apply_final_fixed_threshold(val_prob)
    fold_metrics = {
        "validation": base.evaluate(val_dec, "final_pred", f"fold{fold}_validation")
    }

    for path in test_csvs:
        tag = Path(path).stem
        resid = base.load_residual_seq(path, fc, f"fold{fold}_eval_{tag}")
        feat = base.extract_feat_cached(resid, f"fold{fold}_eval_{tag}")
        feat = feat[feat["ghi_zone"] != "ignore"].reset_index(drop=True)
        feat = feat.drop(columns=drop, errors="ignore")
        feat = base.predict_prob(feat, models, zfc)
        feat = base.apply_final_fixed_threshold(feat)
        fold_metrics[tag] = base.evaluate(feat, "final_pred", tag)
        base.pm(f"fold{fold}_{tag}", fold_metrics[tag])

    return fold_metrics


def main():
    base.ensure_dir(OUT_DIR)
    base.ensure_dir(CACHE_DIR)

    all_rows = []
    all_metrics = {}
    for fold, test_site in enumerate(SITES, start=1):
        fold_metrics = run_fold(fold, test_site)
        all_metrics[f"fold{fold}_{test_site}"] = fold_metrics
        for dataset, metric in fold_metrics.items():
            all_rows.append(metric_row(fold, test_site, dataset, metric))

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = os.path.join(OUT_DIR, "site_kfold_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    test_df = metrics_df[metrics_df["dataset"] != "validation"].copy()
    test_df["attack_ratio_pct"] = test_df["dataset"].str.extract(r"_sa_(\d+)pct").astype(int)

    summary = (
        test_df
        .groupby("attack_ratio_pct")[["precision", "recall", "f1", "fpr", "day_recall", "day_fpr"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = os.path.join(OUT_DIR, "site_kfold_summary_by_attack_ratio.csv")
    summary.to_csv(summary_path, index=False)

    with open(os.path.join(OUT_DIR, "site_kfold_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("K-FOLD SUMMARY BY ATTACK RATIO")
    print("=" * 90)
    print(summary.to_string(index=False))
    print(f"\nsaved: {metrics_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
