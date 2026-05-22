#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost depth7 + tuned Tweedie GBM ensemble detector runner.

경로 설명
---------
- 기준 detector 스크립트: ./test_ver7_catboost_depth7_lgbm_ensemble.py
- CatBoost 모델: ./model_output_catboost_depth_only/cat_depth_7/
- tuned GBM 모델: ./model_output_tweedie_tuning/p12_lr003_l31/
- 비중별 detector 출력:
  ./detector_model_fadre_v3_depth7_lgbm_ensemble_gbmXX_catYY/
- 최종 요약 CSV:
  ./depth7_lgbm_ensemble_detector_summary.csv
  ./depth7_lgbm_ensemble_vs_depth7_comparison.csv

실험 비중
---------
- CatBoost 비중: 0.50, 0.55, 0.60, 0.65, 0.70
- GBM 비중: 1 - CatBoost 비중
"""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "conda_env_solar_tweedie" / "bin" / "python"
SCRIPT = ROOT / "test_ver7_catboost_depth7_lgbm_ensemble.py"
CAT_WEIGHTS = [0.50, 0.55, 0.60, 0.65, 0.70]


def tag_for(cat_weight):
    gbm_weight = 1.0 - cat_weight
    return f"gbm{gbm_weight:.2f}_cat{cat_weight:.2f}".replace(".", "p")


def out_dir_for(cat_weight):
    return ROOT / f"detector_model_fadre_v3_depth7_lgbm_ensemble_{tag_for(cat_weight)}"


def load_metrics(out_dir):
    metrics_path = out_dir / "metrics_iter.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for dataset, values in metrics.items():
        rows.append({"dataset": dataset, **values})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "detector_metrics.csv", index=False)
    eval_df = df[df["dataset"].str.contains("attack_sa_", na=False)].copy()
    eval_df["attack_ratio"] = eval_df["dataset"].str.extract(r"sa_(\d+)pct")[0].astype(int)
    return eval_df


def summarize(eval_df, cat_weight):
    cols = ["precision", "recall", "f1", "fpr", "day_recall", "day_fpr"]
    summary = eval_df.groupby("attack_ratio")[cols].mean().reset_index()
    summary.insert(0, "gbm_weight", round(1.0 - cat_weight, 2))
    summary.insert(1, "cat_weight", round(cat_weight, 2))
    summary.insert(2, "model", f"gbm{1.0 - cat_weight:.2f}_cat{cat_weight:.2f}")
    return summary


def run_one(cat_weight):
    out_dir = out_dir_for(cat_weight)
    env = os.environ.copy()
    env.update(
        {
            "SOLAR_ENSEMBLE_CAT_WEIGHT": str(cat_weight),
            "MPLCONFIGDIR": "/tmp/mpl_solar_depth7_lgbm_ensemble",
            "PYTHONUNBUFFERED": "1",
        }
    )
    print("\n" + "=" * 90, flush=True)
    print(
        f"RUN ensemble | GBM={1.0 - cat_weight:.2f}, CatBoost={cat_weight:.2f} | {out_dir.name}",
        flush=True,
    )
    print("=" * 90, flush=True)
    subprocess.run([str(PYTHON), str(SCRIPT)], cwd=str(ROOT), env=env, check=True)
    return summarize(load_metrics(out_dir), cat_weight)


def main():
    summaries = []
    for cat_weight in CAT_WEIGHTS:
        summaries.append(run_one(cat_weight))
        partial = pd.concat(summaries, ignore_index=True)
        partial.to_csv(ROOT / "depth7_lgbm_ensemble_detector_summary_partial.csv", index=False)
        print("\nPARTIAL SUMMARY")
        print(partial.to_string(index=False))

    ensemble_summary = pd.concat(summaries, ignore_index=True)
    ensemble_summary.to_csv(ROOT / "depth7_lgbm_ensemble_detector_summary.csv", index=False)

    base = pd.read_csv(ROOT / "detector_model_fadre_v3_catboost_depth7" / "detector_catboost_depth7_metrics.csv")
    base_eval = base[base["dataset"].str.contains("attack_sa_", na=False)].copy()
    base_eval["attack_ratio"] = base_eval["dataset"].str.extract(r"sa_(\d+)pct")[0].astype(int)
    base_summary = base_eval.groupby("attack_ratio")[
        ["precision", "recall", "f1", "fpr", "day_recall", "day_fpr"]
    ].mean().reset_index()
    base_summary.insert(0, "gbm_weight", 0.0)
    base_summary.insert(1, "cat_weight", 1.0)
    base_summary.insert(2, "model", "catboost_depth7_only")

    comparison = pd.concat([base_summary, ensemble_summary], ignore_index=True)
    comparison.to_csv(ROOT / "depth7_lgbm_ensemble_vs_depth7_comparison.csv", index=False)

    print("\n" + "=" * 90)
    print("FINAL ENSEMBLE SUMMARY")
    print("=" * 90)
    print(ensemble_summary.to_string(index=False))
    print("\nsaved: depth7_lgbm_ensemble_detector_summary.csv")
    print("saved: depth7_lgbm_ensemble_vs_depth7_comparison.csv")


if __name__ == "__main__":
    main()
