#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost depth7 forecaster + detector within-kW 6:2:2 time split experiment.

경로 설명
---------
- 재사용 코드: ./test_ver7_catboost_depth7.py
- 입력 공격 CSV: ./generated_attack_data/site5_*_attack_sa_{5,8,10}pct.csv
- 출력 폴더: ./detector_model_fadre_v3_catboost_depth7_within_kw_622
- 캐시 폴더: ./feature_cache_fadre_v3_catboost_depth7_within_kw_622

실험 설계
---------
- 각 kW/site별 데이터를 시간순으로 60%/20%/20%로 나눈다.
- detector 학습: 모든 kW/site의 앞 60%, SA 5%, SA 10%
- detector 검증: 모든 kW/site의 중간 20%, SA 5%, SA 10%
- detector 테스트: 모든 kW/site의 뒤 20%, SA 5%, SA 8%, SA 10%
- SA 8%는 학습/검증에 넣지 않고, 테스트에서만 중간 공격 강도 일반화로 본다.
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

OUT_DIR = "./detector_model_fadre_v3_catboost_depth7_within_kw_622"
CACHE_DIR = "./feature_cache_fadre_v3_catboost_depth7_within_kw_622"


def attack_csv(site, pct):
    return f"./generated_attack_data/{site}_attack_sa_{pct}pct.csv"


def split_by_date_622(df):
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    train_end = int(n_dates * 0.60)
    val_end = int(n_dates * 0.80)
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])
    return (
        df[df["date"].isin(train_dates)].reset_index(drop=True),
        df[df["date"].isin(val_dates)].reset_index(drop=True),
        df[df["date"].isin(test_dates)].reset_index(drop=True),
    )


def load_feature_frame(fc, site, pct, tag_prefix):
    path = attack_csv(site, pct)
    tag = Path(path).stem
    resid = base.load_residual_seq(path, fc, f"{tag_prefix}_{tag}")
    feat = base.extract_feat_cached(resid, f"{tag_prefix}_{tag}")
    feat = feat[feat["ghi_zone"] != "ignore"].reset_index(drop=True)
    feat["source_site"] = site
    feat["attack_ratio_pct"] = pct
    return feat


def metric_row(split, site, pct, dataset, metric):
    row = {
        "split": split,
        "site": site,
        "attack_ratio_pct": pct,
        "dataset": dataset,
    }
    row.update(metric)
    return row


def main():
    base.OUT_DIR = OUT_DIR
    base.CACHE_DIR = CACHE_DIR
    base.ensure_dir(OUT_DIR)
    base.ensure_dir(CACHE_DIR)

    fc = base.ForecasterWrapper().load()
    base.log("Residual centering/calibration disabled: using raw forecaster predictions only")

    train_parts = []
    val_parts = []
    test_parts = []
    split_rows = []

    for site in SITES:
        for pct in TRAIN_ATTACK_PCTS:
            feat = load_feature_frame(fc, site, pct, "within622_trainval")
            dtr, dva, dte = split_by_date_622(feat)
            train_parts.append(dtr)
            val_parts.append(dva)
            split_rows.append(
                {
                    "site": site,
                    "attack_ratio_pct": pct,
                    "role": "train_source",
                    "train_rows": len(dtr),
                    "val_rows": len(dva),
                    "heldout_test_rows": len(dte),
                    "train_pos": int(dtr["is_attack"].sum()),
                    "val_pos": int(dva["is_attack"].sum()),
                    "heldout_test_pos": int(dte["is_attack"].sum()),
                }
            )

        for pct in TEST_ATTACK_PCTS:
            feat = load_feature_frame(fc, site, pct, "within622_test")
            _, _, dte = split_by_date_622(feat)
            test_parts.append((site, pct, Path(attack_csv(site, pct)).stem, dte))

    dtr = pd.concat(train_parts, ignore_index=True)
    dva = pd.concat(val_parts, ignore_index=True)
    base.log(
        f"within-kw 6:2:2 train={len(dtr):,}(pos={int(dtr['is_attack'].sum())}) "
        f"val={len(dva):,}(pos={int(dva['is_attack'].sum())})"
    )

    zfc, drop = base.select_zone_features(dtr)
    dtr = dtr.drop(columns=drop, errors="ignore")
    dva = dva.drop(columns=drop, errors="ignore")
    base.log(
        f"within-kw 6:2:2 dropped {len(drop)} common features | "
        f"high={len(zfc['high'])} mid={len(zfc['mid'])}"
    )

    models, val_prob = base.train_zone_models(dtr, dva, zfc)
    val_dec = base.apply_final_fixed_threshold(val_prob)

    all_rows = [metric_row("validation", "all", -1, "validation", base.evaluate(val_dec, "final_pred", "validation"))]
    all_metrics = {"validation": all_rows[0]}

    for site, pct, tag, feat in test_parts:
        feat = feat.drop(columns=drop, errors="ignore")
        feat = base.predict_prob(feat, models, zfc)
        feat = base.apply_final_fixed_threshold(feat)
        metric = base.evaluate(feat, "final_pred", tag)
        all_metrics[tag] = metric
        all_rows.append(metric_row("test", site, pct, tag, metric))
        base.pm(f"within622_{tag}", metric)

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = os.path.join(OUT_DIR, "within_kw_622_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    split_df = pd.DataFrame(split_rows)
    split_path = os.path.join(OUT_DIR, "within_kw_622_split_counts.csv")
    split_df.to_csv(split_path, index=False)

    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    summary = (
        test_df
        .groupby("attack_ratio_pct")[["precision", "recall", "f1", "fpr", "day_recall", "day_fpr"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = os.path.join(OUT_DIR, "within_kw_622_summary_by_attack_ratio.csv")
    summary.to_csv(summary_path, index=False)

    with open(os.path.join(OUT_DIR, "within_kw_622_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("WITHIN-KW 6:2:2 SUMMARY BY ATTACK RATIO")
    print("=" * 90)
    print(summary.to_string(index=False))
    print(f"\nsaved: {metrics_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {split_path}")


if __name__ == "__main__":
    main()
