#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost learning-rate-only tuning runner.

경로 설명
---------
- 기준 학습 스크립트: ./entro2_tweedie.py
- 출력 폴더: ${SOLAR_CAT_LR_OUT_ROOT:-./model_output_catboost_lr_only}/<config_name>/
- 요약 CSV: ${SOLAR_CAT_LR_OUT_ROOT:-./model_output_catboost_lr_only}/catboost_lr_only_summary.csv

실험 목적
---------
- depth=8로 고정하고 learning_rate만 바꾼다.
- 나머지 주요 값은 기본 CatBoost 설정으로 고정한다.
  depth=8, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0
- SOLAR_CAT_TASK_TYPE/SOLAR_CAT_DEVICES 환경변수로 GPU 사용 여부를 제어한다.
"""

import os
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "conda_env_solar_tweedie" / "bin" / "python"
SCRIPT = ROOT / "entro2_tweedie.py"
OUT_ROOT = Path(os.environ.get("SOLAR_CAT_LR_OUT_ROOT", ROOT / "model_output_catboost_lr_only"))
if not OUT_ROOT.is_absolute():
    OUT_ROOT = ROOT / OUT_ROOT

LEARNING_RATES = [0.01, 0.02, 0.03, 0.05, 0.07]

CAT_TEST_RE = re.compile(
    r"=== CatBoost calibrated ===.*?TEST\s*:\s*R2=([0-9.\-]+)\s+MAE=([0-9.\-]+)%\s+RMSE=([0-9.\-]+)%",
    re.S,
)


def lr_name(lr):
    return str(lr).replace(".", "p")


def parse_cat_metrics(path):
    text = Path(path).read_text(encoding="utf-8")
    match = CAT_TEST_RE.search(text)
    if not match:
        raise RuntimeError(f"Could not parse CatBoost TEST metrics from {path}")
    return {
        "test_r2": float(match.group(1)),
        "test_mae_pct": float(match.group(2)),
        "test_rmse_pct": float(match.group(3)),
    }


def run_one(lr):
    name = f"cat_depth8_lr_{lr_name(lr)}"
    out_dir = OUT_ROOT / name
    task_type = os.environ.get("SOLAR_CAT_TASK_TYPE", "CPU")
    devices = os.environ.get("SOLAR_CAT_DEVICES", "0")
    env = os.environ.copy()
    env.update(
        {
            "SOLAR_TWEEDIE_OUT_DIR": str(out_dir),
            "SOLAR_LGBM_N_ESTIMATORS": "10",
            "SOLAR_RUN_CATBOOST": "1",
            "SOLAR_CAT_ITERATIONS": "8000",
            "SOLAR_CAT_LEARNING_RATE": str(lr),
            "SOLAR_CAT_DEPTH": "8",
            "SOLAR_CAT_L2_LEAF_REG": "3.0",
            "SOLAR_CAT_RANDOM_STRENGTH": "1.0",
            "SOLAR_CAT_BAGGING_TEMPERATURE": "1.0",
            "SOLAR_CAT_TASK_TYPE": task_type,
            "SOLAR_CAT_DEVICES": devices,
            "MPLCONFIGDIR": "/tmp/mpl_solar_catboost_lr_only",
            "PYTHONUNBUFFERED": "1",
        }
    )

    print("\n" + "=" * 90, flush=True)
    print(f"RUN {name} | depth=8, lr={lr}, l2=3.0, task={task_type}, devices={devices}", flush=True)
    print("=" * 90, flush=True)
    subprocess.run([str(PYTHON), str(SCRIPT)], cwd=str(ROOT), env=env, check=True)
    return {
        "name": name,
        "depth": 8,
        "learning_rate": lr,
        "task_type": task_type,
        "devices": devices,
        **parse_cat_metrics(out_dir / "metrics_compare.txt"),
        "out_dir": str(out_dir),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lr in LEARNING_RATES:
        rows.append(run_one(lr))
        partial = pd.DataFrame(rows).sort_values(["test_mae_pct", "test_rmse_pct"])
        partial.to_csv(OUT_ROOT / "catboost_lr_only_summary_partial.csv", index=False)
        print("\nPARTIAL SUMMARY")
        print(partial.to_string(index=False))

    summary = pd.DataFrame(rows).sort_values(["test_mae_pct", "test_rmse_pct"])
    out_path = OUT_ROOT / "catboost_lr_only_summary.csv"
    summary.to_csv(out_path, index=False)
    print("\n" + "=" * 90)
    print("FINAL CATBOOST LR-ONLY SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
