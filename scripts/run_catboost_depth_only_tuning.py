#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost depth-only tuning runner.

경로 설명
---------
- 기준 학습 스크립트: ./entro2_tweedie.py
- 출력 폴더: ./model_output_catboost_depth_only/<config_name>/
- 요약 CSV: ./model_output_catboost_depth_only/catboost_depth_only_summary.csv
- 특정 depth만 추가 실행하려면 SOLAR_CAT_DEPTHS=7,9 처럼 지정한다.

실험 목적
---------
- CatBoost에서 가장 영향이 큰 파라미터로 알려진 depth만 바꾼다.
- 나머지 값은 기본 CatBoost 설정으로 고정한다.
  learning_rate=0.03, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0
"""

import os
import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "conda_env_solar_tweedie" / "bin" / "python"
SCRIPT = ROOT / "entro2_tweedie.py"
OUT_ROOT = ROOT / "model_output_catboost_depth_only"

DEPTHS = [int(x.strip()) for x in os.environ.get("SOLAR_CAT_DEPTHS", "4,6,8,10").split(",") if x.strip()]

CAT_TEST_RE = re.compile(
    r"=== CatBoost calibrated ===.*?TEST\s*:\s*R2=([0-9.\-]+)\s+MAE=([0-9.\-]+)%\s+RMSE=([0-9.\-]+)%",
    re.S,
)


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


def run_one(depth):
    name = f"cat_depth_{depth}"
    out_dir = OUT_ROOT / name
    env = os.environ.copy()
    env.update(
        {
            "SOLAR_TWEEDIE_OUT_DIR": str(out_dir),
            "SOLAR_LGBM_N_ESTIMATORS": "10",
            "SOLAR_RUN_CATBOOST": "1",
            "SOLAR_CAT_ITERATIONS": "8000",
            "SOLAR_CAT_LEARNING_RATE": "0.03",
            "SOLAR_CAT_DEPTH": str(depth),
            "SOLAR_CAT_L2_LEAF_REG": "3.0",
            "SOLAR_CAT_RANDOM_STRENGTH": "1.0",
            "SOLAR_CAT_BAGGING_TEMPERATURE": "1.0",
            "MPLCONFIGDIR": "/tmp/mpl_solar_catboost_depth_only",
            "PYTHONUNBUFFERED": "1",
        }
    )

    print("\n" + "=" * 90, flush=True)
    print(f"RUN {name} | depth={depth}, lr=0.03, l2=3.0", flush=True)
    print("=" * 90, flush=True)
    subprocess.run([str(PYTHON), str(SCRIPT)], cwd=str(ROOT), env=env, check=True)
    return {"name": name, "depth": depth, **parse_cat_metrics(out_dir / "metrics_compare.txt"), "out_dir": str(out_dir)}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in DEPTHS:
        rows.append(run_one(depth))
        partial = pd.DataFrame(rows).sort_values(["test_mae_pct", "test_rmse_pct"])
        partial.to_csv(OUT_ROOT / "catboost_depth_only_summary_partial.csv", index=False)
        print("\nPARTIAL SUMMARY")
        print(partial.to_string(index=False))

    out_path = OUT_ROOT / "catboost_depth_only_summary.csv"
    new_summary = pd.DataFrame(rows)
    if out_path.exists():
        old_summary = pd.read_csv(out_path)
        summary = pd.concat([old_summary, new_summary], ignore_index=True)
        summary = summary.drop_duplicates(subset=["name", "depth"], keep="last")
    else:
        summary = new_summary
    summary = summary.sort_values(["test_mae_pct", "test_rmse_pct"])
    summary.to_csv(out_path, index=False)
    print("\n" + "=" * 90)
    print("FINAL CATBOOST DEPTH-ONLY SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
