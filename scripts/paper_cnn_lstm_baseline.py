#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-style CNN-LSTM probabilistic anomaly detection baseline.

이 스크립트는 Zhang et al. (2022)의 "CNN-LSTM deterministic forecast
-> Gaussian prediction interval -> anomaly detection" 흐름을 현재 제공된
4-site PV clean 데이터에 맞춰 재현하기 위한 baseline 구현입니다.

중요한 경로 설정
----------------
1) 입력 데이터 zip
   기본값: ./data-20260521T052311Z-3-001.zip
   zip 내부에는 아래 4개 clean CSV가 있어야 합니다.

   data/site5_5.9kw_2016_2019_clean.csv
   data/site5_7.0kw_2016_2019_clean.csv
   data/site5_327.6kw_2016_2019_clean.csv
   data/site5_226.8kw_2016_2019_clean.csv

   만약 zip을 풀어둔 경우에는 --data-dir ./data 로도 읽을 수 있습니다.
   같은 파일명이 data-dir 안에 있으면 data-dir을 우선 사용하고,
   없으면 zip에서 직접 읽습니다.

2) 출력 경로
   기본값: ./paper_cnn_lstm_outputs
   생성 파일:
   - metrics_summary.csv: clean/SA 5/8/10% 탐지 성능 요약
   - predictions_clean.csv: clean test prediction interval 결과
   - predictions_attack_sa_5pct.csv
   - predictions_attack_sa_8pct.csv
   - predictions_attack_sa_10pct.csv
   - model_state.pt: 학습된 CNN-LSTM state_dict 및 설정
   - run_config.json: 실행 설정

3) 기본 실행
   python3 paper_cnn_lstm_baseline.py

4) 빠른 코드 검증 실행
   python3 paper_cnn_lstm_baseline.py --quick --epochs 1

5) 논문 해상도에 더 가깝게 30분 단위로 맞춰 실행
   python3 paper_cnn_lstm_baseline.py --resample-rule 30min --seq-len 8 --epochs 30

실험 설계 메모
--------------
- 원 논문 데이터는 공개되어 있지 않으므로, 여기서는 "동일 데이터 재현"이 아니라
  "방법론 재현"을 수행합니다.
- 입력 feature는 요청대로 [과거 발전량, 일사량, 기온]만 사용합니다.
  구체적으로 각 site/time step마다 [power_ratio, ghi, temp] 3개를 사용합니다.
- 4개 site는 2 x 2 spatial matrix로 배치합니다.
- 기본값은 원본 5분 해상도를 유지하고 1-hour-ahead = 12-step ahead로 예측합니다.
  --resample-rule 30min 을 주면 1-hour-ahead = 2-step ahead가 됩니다.
- 공격 데이터는 clean test period에 plateau-type scaling attack을 합성합니다.
  SA 5%, 8%, 10%는 같은 attack mask를 공유하고 delta만 다르게 적용합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLEAN_CSVS = [
    "site5_5.9kw_2016_2019_clean.csv",
    "site5_7.0kw_2016_2019_clean.csv",
    "site5_327.6kw_2016_2019_clean.csv",
    "site5_226.8kw_2016_2019_clean.csv",
]

FEATURE_COLUMNS = ["power_ratio", "ghi", "temp"]
TARGET_COLUMN = "power_ratio"


@dataclass
class RunConfig:
    data_zip: str
    data_dir: str
    output_dir: str
    seq_len: int
    horizon_minutes: int
    horizon_steps: int
    resample_rule: str
    epochs: int
    batch_size: int
    learning_rate: float
    confidence: float
    attack_deltas: List[float]
    attack_ratio: float
    min_ghi_attack: float
    quick: bool
    max_samples_per_split: int
    sigma_sample_limit: int
    seed: int
    device: str


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def log(self, msg: str) -> None:
        elapsed = time.time() - self.start
        print(f"[+{elapsed:7.1f}s] {msg}", flush=True)


class CnnLstmForecaster(nn.Module):
    """Small CNN-LSTM adapted from the paper for a 2 x 2 site matrix."""

    def __init__(self, channels: int = 3, height: int = 2, width: int = 2) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 50, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(50 * (height // 2) * (width // 2), 50),
            nn.ReLU(),
        )
        self.lstm1 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.10)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=100, batch_first=True)
        self.head = nn.Linear(100, height * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, channels, height, width]
        batch, seq_len, channels, height, width = x.shape
        z = x.reshape(batch * seq_len, channels, height, width)
        z = self.cnn(z)
        z = z.reshape(batch, seq_len, -1)
        z, _ = self.lstm1(z)
        z = self.dropout(z)
        z, _ = self.lstm2(z)
        return self.head(z[:, -1, :])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-style CNN-LSTM probabilistic anomaly detection baseline"
    )
    parser.add_argument("--data-zip", default="data-20260521T052311Z-3-001.zip")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="paper_cnn_lstm_outputs")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--resample-rule", default="", help="Example: 30min. Empty keeps native data.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--attack-deltas", type=float, nargs="+", default=[0.05, 0.08, 0.10])
    parser.add_argument("--attack-ratio", type=float, default=0.045)
    parser.add_argument("--min-ghi-attack", type=float, default=200.0)
    parser.add_argument("--quick", action="store_true", help="Use fewer samples for a fast smoke run.")
    parser.add_argument("--max-samples-per-split", type=int, default=3000)
    parser.add_argument("--sigma-sample-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_clean_csv(csv_name: str, data_dir: Path, data_zip: Path) -> pd.DataFrame:
    path = data_dir / csv_name
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")

    internal = f"data/{csv_name}"
    if not data_zip.exists():
        raise FileNotFoundError(f"Missing both {path} and {data_zip}")
    with zipfile.ZipFile(data_zip) as zf:
        if internal not in zf.namelist():
            raise FileNotFoundError(f"Missing {internal} in {data_zip}")
        with zf.open(internal) as fh:
            return pd.read_csv(fh, encoding="utf-8-sig")


def clean_site_frame(df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first")

    required = ["timestamp", "site", "capacity_kw", "power", "power_ratio", "ghi", "temp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in ["capacity_kw", "power", "power_ratio", "ghi", "temp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # The provided large-site CSVs contain a few sentinel values around -8999999488.
    df.loc[(df["temp"] < -50) | (df["temp"] > 80), "temp"] = np.nan
    df.loc[(df["ghi"] < 0) | (df["ghi"] > 1600), "ghi"] = np.nan
    df.loc[(df["power_ratio"] < 0) | (df["power_ratio"] > 1.5), "power_ratio"] = np.nan

    site = str(df["site"].dropna().iloc[0])
    capacity = float(df["capacity_kw"].dropna().iloc[0])

    df = df.set_index("timestamp")
    numeric = df[["capacity_kw", "power", "power_ratio", "ghi", "temp"]].copy()
    if resample_rule:
        numeric = numeric.resample(resample_rule).mean()

    numeric.loc[:, ["power", "power_ratio", "ghi", "temp"]] = numeric[
        ["power", "power_ratio", "ghi", "temp"]
    ].interpolate(method="time", limit=3)
    numeric.loc[:, "capacity_kw"] = capacity
    numeric = numeric.dropna(subset=["power", "power_ratio", "ghi", "temp"])
    numeric["site"] = site
    return numeric.reset_index()


def load_all_sites(args: argparse.Namespace, timer: Timer) -> List[pd.DataFrame]:
    data_dir = Path(args.data_dir)
    data_zip = Path(args.data_zip)
    frames = []
    for name in CLEAN_CSVS:
        df = read_clean_csv(name, data_dir, data_zip)
        df = clean_site_frame(df, args.resample_rule)
        timer.log(
            f"loaded {name}: rows={len(df):,}, "
            f"{df['timestamp'].min()} -> {df['timestamp'].max()}"
        )
        frames.append(df)
    return frames


def build_aligned_arrays(
    frames: Sequence[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    sites = [str(df["site"].iloc[0]) for df in frames]
    common = set(frames[0]["timestamp"])
    for df in frames[1:]:
        common &= set(df["timestamp"])
    timestamps = pd.Index(sorted(common))
    if len(timestamps) == 0:
        raise ValueError("No common timestamps across sites.")

    feature_blocks = []
    target_blocks = []
    capacities = []
    for df in frames:
        site = str(df["site"].iloc[0])
        aligned = df.set_index("timestamp").loc[timestamps]
        feature_blocks.append(aligned[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        target_blocks.append(aligned[TARGET_COLUMN].to_numpy(dtype=np.float32))
        capacities.append(float(aligned["capacity_kw"].iloc[0]))

    # [time, site, feature] -> [time, channel, height, width]
    features_site = np.stack(feature_blocks, axis=1)
    targets_site = np.stack(target_blocks, axis=1)
    features = features_site.reshape(len(timestamps), 2, 2, len(FEATURE_COLUMNS))
    features = np.transpose(features, (0, 3, 1, 2)).astype(np.float32)
    targets = targets_site.astype(np.float32)
    return timestamps.to_numpy(), features, sites, targets, np.asarray(capacities)


def infer_step_minutes(timestamps: np.ndarray) -> int:
    ts = pd.to_datetime(timestamps)
    diffs = pd.Series(ts).diff().dropna()
    if diffs.empty:
        raise ValueError("Need at least two timestamps.")
    return int(round(diffs.median().total_seconds() / 60.0))


def valid_sample_indices(
    timestamps: np.ndarray,
    seq_len: int,
    horizon_steps: int,
) -> np.ndarray:
    ts_ns = pd.to_datetime(timestamps).astype("int64").to_numpy()
    step_ns = int(np.median(np.diff(ts_ns)))
    contiguous = (np.diff(ts_ns) == step_ns).astype(np.int32)
    needed_edges = seq_len - 1 + horizon_steps
    prefix = np.concatenate([[0], np.cumsum(contiguous)])
    starts = np.arange(0, len(timestamps) - needed_edges, dtype=np.int64)
    good = (prefix[starts + needed_edges] - prefix[starts]) == needed_edges
    endpoints = starts[good] + seq_len - 1
    return endpoints.astype(np.int64)


def build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    endpoints: np.ndarray,
    seq_len: int,
    horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(endpoints)
    channels, height, width = features.shape[1:]
    x = np.empty((n, seq_len, channels, height, width), dtype=np.float32)
    y = np.empty((n, height * width), dtype=np.float32)
    target_indices = endpoints + horizon_steps
    for row, endpoint in enumerate(endpoints):
        start = endpoint - seq_len + 1
        x[row] = features[start : endpoint + 1]
        y[row] = targets[target_indices[row]]
    return x, y, target_indices


def split_indices(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Paper split: first 3/4 train, last 1/4 test.
    # Within train: first 2/3 deterministic model, last 1/3 probabilistic calibration.
    n_det = int(n * 0.50)
    n_cal = int(n * 0.75)
    return np.arange(0, n_det), np.arange(n_det, n_cal), np.arange(n_cal, n)


def limit_split(idx: np.ndarray, max_count: int) -> np.ndarray:
    if len(idx) <= max_count:
        return idx
    return idx[:max_count]


def fit_feature_scaler(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1, 3, 4), keepdims=True)
    std = x_train.std(axis=(0, 1, 3, 4), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def train_model(
    model: CnnLstmForecaster,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    timer: Timer,
) -> CnnLstmForecaster:
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).detach().cpu())
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        timer.log(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_mse={np.mean(losses):.6f} val_mse={val_loss:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_model(
    model: CnnLstmForecaster,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(device)).detach().cpu().numpy()
            preds.append(pred)
    return np.clip(np.concatenate(preds, axis=0), 0.0, 1.2).astype(np.float32)


def pinball_sigma_multiplier() -> float:
    taus = np.arange(1, 100, dtype=np.float64) / 100.0
    z = norm.ppf(taus)
    # Because pinball loss scales with abs residual, optimize for residual=1.
    candidates = np.linspace(0.01, 5.0, 3000)
    q = candidates[:, None] * z[None, :]
    y = 1.0
    loss = np.where(y < q, (1.0 - taus) * (q - y), taus * (y - q)).mean(axis=1)
    return float(candidates[int(np.argmin(loss))])


def fit_sigma_models(
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    sample_limit: int,
    seed: int,
) -> Tuple[List[object], float]:
    multiplier = pinball_sigma_multiplier()
    sigma_star = np.clip(multiplier * np.abs(y_cal - pred_cal), 0.005, 0.50)
    rng = np.random.default_rng(seed)
    models = []
    for site_idx in range(pred_cal.shape[1]):
        x = pred_cal[:, [site_idx]]
        y = sigma_star[:, site_idx]
        if len(x) > sample_limit:
            take = rng.choice(len(x), size=sample_limit, replace=False)
            x_fit = x[take]
            y_fit = y[take]
        else:
            x_fit = x
            y_fit = y
        model = make_pipeline(
            StandardScaler(),
            SVR(C=10.0, epsilon=0.005, gamma="scale"),
        )
        model.fit(x_fit, y_fit)
        models.append(model)
    return models, multiplier


def predict_sigma(models: Sequence[object], pred: np.ndarray) -> np.ndarray:
    sigmas = []
    for site_idx, model in enumerate(models):
        s = model.predict(pred[:, [site_idx]])
        sigmas.append(s)
    sigma = np.stack(sigmas, axis=1)
    return np.clip(sigma, 0.005, 0.50).astype(np.float32)


def calibrate_sigma_factor(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    sigma_cal: np.ndarray,
    confidence: float,
) -> Tuple[float, float, float]:
    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))
    raw_inside = np.abs(y_cal - pred_cal) <= (z * sigma_cal)
    raw_coverage = float(raw_inside.mean())
    ratio = np.abs(y_cal - pred_cal) / (z * np.maximum(sigma_cal, 1e-6))
    factor = float(np.quantile(ratio.reshape(-1), confidence))
    factor = float(np.clip(factor, 0.50, 5.00))
    calibrated_inside = np.abs(y_cal - pred_cal) <= (z * sigma_cal * factor)
    calibrated_coverage = float(calibrated_inside.mean())
    return factor, raw_coverage, calibrated_coverage


def generate_attack_mask(
    timestamps: np.ndarray,
    ghi_site: np.ndarray,
    step_minutes: int,
    target_ratio: float,
    min_ghi: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ts = pd.to_datetime(timestamps)
    n_time, n_site = ghi_site.shape
    mask = np.zeros((n_time, n_site), dtype=bool)
    duration_minutes = np.array([30, 60, 120, 180])
    duration_points = np.maximum(1, np.round(duration_minutes / step_minutes).astype(int))

    for site_idx in range(n_site):
        valid = ghi_site[:, site_idx] >= min_ghi
        target = max(1, int(valid.sum() * target_ratio))
        attempts = 0
        while mask[:, site_idx].sum() < target and attempts < target * 50:
            attempts += 1
            length = int(rng.choice(duration_points))
            if length >= n_time:
                continue
            start = int(rng.integers(0, n_time - length))
            end = start + length
            if not valid[start:end].all():
                continue
            if ts[start].date() != ts[end - 1].date():
                continue
            diffs = np.diff(ts[start:end].astype("int64"))
            if len(diffs) and not np.all(diffs == int(step_minutes * 60 * 1e9)):
                continue
            if mask[start:end, site_idx].any():
                continue
            mask[start:end, site_idx] = True
    return mask


def apply_scaling_attack(
    clean_features: np.ndarray,
    clean_targets: np.ndarray,
    attack_mask: np.ndarray,
    delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    attacked_targets = clean_targets.copy()
    attacked_targets[attack_mask] = np.minimum(attacked_targets[attack_mask] * (1.0 + delta), 1.2)

    attacked_features = clean_features.copy()
    power_matrix = attacked_targets.reshape(attacked_targets.shape[0], 2, 2)
    attacked_features[:, 0, :, :] = power_matrix
    return attacked_features, attacked_targets


def interval_predictions(
    pred: np.ndarray,
    sigma: np.ndarray,
    confidence: float,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - confidence
    z = float(norm.ppf(1.0 - alpha / 2.0))
    lower = np.clip(pred - z * sigma, 0.0, 1.2)
    upper = np.clip(pred + z * sigma, 0.0, 1.2)
    return lower.astype(np.float32), upper.astype(np.float32)


def compute_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    label: np.ndarray,
    target_timestamps: np.ndarray,
) -> Dict[str, float]:
    pred_anom = ((y_true < lower) | (y_true > upper)).astype(int)
    y_label = label.astype(int)
    cm = confusion_matrix(y_label.reshape(-1), pred_anom.reshape(-1), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    day_df = pd.DataFrame(
        {
            "date": pd.to_datetime(target_timestamps).date,
            "has_attack": y_label.max(axis=1),
            "has_detection": pred_anom.max(axis=1),
        }
    )
    by_day = day_df.groupby("date").max()
    attack_days = by_day[by_day["has_attack"] == 1]
    clean_days = by_day[by_day["has_attack"] == 0]
    day_recall = (
        float((attack_days["has_detection"] == 1).mean()) if len(attack_days) else 0.0
    )
    day_fpr = float((clean_days["has_detection"] == 1).mean()) if len(clean_days) else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall_tpr": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
        "day_recall": float(day_recall),
        "day_fpr": float(day_fpr),
        "attack_points": int(y_label.sum()),
        "detected_points": int(pred_anom.sum()),
    }


def forecast_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - y_true
    return {
        "mae_a": float(np.mean(np.abs(err))),
        "rmse_a": float(np.sqrt(np.mean(err * err))),
    }


def flatten_prediction_frame(
    name: str,
    timestamps: np.ndarray,
    sites: Sequence[str],
    y_true: np.ndarray,
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    pred_anom = ((y_true < lower) | (y_true > upper)).astype(int)
    for site_idx, site in enumerate(sites):
        rows.append(
            pd.DataFrame(
                {
                    "scenario": name,
                    "timestamp": timestamps,
                    "site": site,
                    "y_true": y_true[:, site_idx],
                    "y_pred": pred[:, site_idx],
                    "pi_lower": lower[:, site_idx],
                    "pi_upper": upper[:, site_idx],
                    "is_attack": labels[:, site_idx].astype(int),
                    "is_anomaly": pred_anom[:, site_idx].astype(int),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def save_model_state(
    output_dir: Path,
    model: CnnLstmForecaster,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    config: RunConfig,
    sites: Sequence[str],
    sigma_multiplier: float,
    sigma_calibration_factor: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "config": asdict(config),
            "sites": list(sites),
            "sigma_multiplier": sigma_multiplier,
            "sigma_calibration_factor": sigma_calibration_factor,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        output_dir / "model_state.pt",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    timer = Timer()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer.log(f"device={device}")

    frames = load_all_sites(args, timer)
    timestamps, clean_features, sites, clean_targets, capacities = build_aligned_arrays(frames)
    step_minutes = infer_step_minutes(timestamps)
    horizon_steps = max(1, int(round(args.horizon_minutes / step_minutes)))
    endpoints = valid_sample_indices(timestamps, args.seq_len, horizon_steps)
    if len(endpoints) < 100:
        raise ValueError(f"Too few valid contiguous samples: {len(endpoints)}")

    timer.log(
        f"aligned timestamps={len(timestamps):,}, sites={sites}, "
        f"step={step_minutes}min, horizon_steps={horizon_steps}, "
        f"valid_samples={len(endpoints):,}"
    )

    x_all, y_all, target_indices = build_sequences(
        clean_features, clean_targets, endpoints, args.seq_len, horizon_steps
    )
    idx_det, idx_cal, idx_test = split_indices(len(x_all))
    if args.quick:
        idx_det = limit_split(idx_det, args.max_samples_per_split)
        idx_cal = limit_split(idx_cal, args.max_samples_per_split)
        idx_test = limit_split(idx_test, args.max_samples_per_split)
        timer.log(
            f"quick mode: det={len(idx_det):,}, cal={len(idx_cal):,}, test={len(idx_test):,}"
        )

    x_det_raw, y_det = x_all[idx_det], y_all[idx_det]
    x_cal_raw, y_cal = x_all[idx_cal], y_all[idx_cal]
    x_test_raw, y_test_clean = x_all[idx_test], y_all[idx_test]
    test_target_indices = target_indices[idx_test]
    test_timestamps = timestamps[test_target_indices]

    feature_mean, feature_std = fit_feature_scaler(x_det_raw)
    x_det = apply_feature_scaler(x_det_raw, feature_mean, feature_std)
    x_cal = apply_feature_scaler(x_cal_raw, feature_mean, feature_std)
    x_test_clean = apply_feature_scaler(x_test_raw, feature_mean, feature_std)

    model = CnnLstmForecaster(channels=len(FEATURE_COLUMNS), height=2, width=2)
    model = train_model(model, x_det, y_det, x_cal, y_cal, args, device, timer)

    pred_cal = predict_model(model, x_cal, args.batch_size, device)
    sigma_models, sigma_multiplier = fit_sigma_models(
        pred_cal, y_cal, args.sigma_sample_limit, args.seed
    )
    sigma_cal_raw = predict_sigma(sigma_models, pred_cal)
    sigma_calibration_factor, raw_coverage, calibrated_coverage = calibrate_sigma_factor(
        y_cal, pred_cal, sigma_cal_raw, args.confidence
    )
    timer.log(
        f"sigma surrogate fitted, pinball_sigma_multiplier={sigma_multiplier:.4f}, "
        f"sigma_calibration_factor={sigma_calibration_factor:.4f}, "
        f"cal_coverage={raw_coverage:.3f}->{calibrated_coverage:.3f}"
    )

    pred_clean = predict_model(model, x_test_clean, args.batch_size, device)
    sigma_clean = predict_sigma(sigma_models, pred_clean) * sigma_calibration_factor
    lower_clean, upper_clean = interval_predictions(pred_clean, sigma_clean, args.confidence)
    clean_labels = np.zeros_like(y_test_clean, dtype=int)
    clean_metrics = compute_metrics(
        y_test_clean, lower_clean, upper_clean, clean_labels, test_timestamps
    )
    clean_metrics.update(forecast_metrics(y_test_clean, pred_clean))
    clean_metrics["scenario"] = "clean"
    clean_metrics["delta"] = 0.0

    flatten_prediction_frame(
        "clean",
        test_timestamps,
        sites,
        y_test_clean,
        pred_clean,
        lower_clean,
        upper_clean,
        clean_labels,
    ).to_csv(output_dir / "predictions_clean.csv", index=False)

    ghi_site = clean_features[:, 1, :, :].reshape(len(clean_features), 4)
    attack_mask_full = generate_attack_mask(
        timestamps,
        ghi_site,
        step_minutes,
        args.attack_ratio,
        args.min_ghi_attack,
        args.seed,
    )
    timer.log(
        "attack mask generated: "
        + ", ".join(
            f"{site}={attack_mask_full[:, i].mean() * 100:.2f}%"
            for i, site in enumerate(sites)
        )
    )

    rows = [clean_metrics]
    for delta in args.attack_deltas:
        attack_features, attack_targets = apply_scaling_attack(
            clean_features, clean_targets, attack_mask_full, delta
        )
        x_attack_raw, y_attack_all, _ = build_sequences(
            attack_features, attack_targets, endpoints, args.seq_len, horizon_steps
        )
        labels_all = attack_mask_full[target_indices]
        x_attack = apply_feature_scaler(x_attack_raw[idx_test], feature_mean, feature_std)
        y_attack = y_attack_all[idx_test]
        labels = labels_all[idx_test].astype(int)

        pred_attack = predict_model(model, x_attack, args.batch_size, device)
        sigma_attack = predict_sigma(sigma_models, pred_attack) * sigma_calibration_factor
        lower_attack, upper_attack = interval_predictions(
            pred_attack, sigma_attack, args.confidence
        )
        metrics = compute_metrics(
            y_attack, lower_attack, upper_attack, labels, test_timestamps
        )
        metrics.update(forecast_metrics(y_attack, pred_attack))
        metrics["scenario"] = f"attack_sa_{int(round(delta * 100))}pct"
        metrics["delta"] = float(delta)
        rows.append(metrics)

        out_name = f"predictions_attack_sa_{int(round(delta * 100))}pct.csv"
        flatten_prediction_frame(
            metrics["scenario"],
            test_timestamps,
            sites,
            y_attack,
            pred_attack,
            lower_attack,
            upper_attack,
            labels,
        ).to_csv(output_dir / out_name, index=False)
        timer.log(
            f"{metrics['scenario']}: F1={metrics['f1']:.4f}, "
            f"TPR={metrics['recall_tpr']:.4f}, FPR={metrics['fpr']:.4f}"
        )

    metrics_df = pd.DataFrame(rows)
    front = ["scenario", "delta", "precision", "recall_tpr", "fpr", "f1", "mae_a", "rmse_a"]
    metrics_df = metrics_df[front + [c for c in metrics_df.columns if c not in front]]
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    config = RunConfig(
        data_zip=args.data_zip,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        horizon_minutes=args.horizon_minutes,
        horizon_steps=horizon_steps,
        resample_rule=args.resample_rule,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        confidence=args.confidence,
        attack_deltas=[float(x) for x in args.attack_deltas],
        attack_ratio=args.attack_ratio,
        min_ghi_attack=args.min_ghi_attack,
        quick=args.quick,
        max_samples_per_split=args.max_samples_per_split,
        sigma_sample_limit=args.sigma_sample_limit,
        seed=args.seed,
        device=str(device),
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, ensure_ascii=False, indent=2)
    save_model_state(
        output_dir,
        model.cpu(),
        feature_mean,
        feature_std,
        config,
        sites,
        sigma_multiplier,
        sigma_calibration_factor,
    )

    timer.log("done")
    print("\n=== metrics_summary ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
