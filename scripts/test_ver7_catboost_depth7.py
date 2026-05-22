#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FADRE v3.11 — full debug tuning integrated
═══════════════════════════════════════════
- DROP_FEATURES = [] 로 시작 (전부 살림)
- 돌리고 importance 보고 → DROP_FEATURES에 추가 → 재실행
- 캐시 버전 분리해서 이전 캐시 안 깨짐
"""

import os
import json
import pickle
import hashlib
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from scipy.stats import skew as scipy_skew, kurtosis

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

warnings.filterwarnings("ignore")

# =============================================================================
# 0. Paths
# =============================================================================
# No residual calibration is used in v3.5. This path is kept only for compatibility/reference.
# Tweedie forecaster detector experiment.
# Train detector on held-in sites/intensities and evaluate on held-out sites.
TRAIN_CLEAN_CSV = "./dataset_clean/site5_5.9kw_2016_2019_clean.csv"
TRAIN_ATTACK_CSVS = [
    "./generated_attack_data/site5_5.9kw_2016_2019_attack_sa_5pct.csv",
    "./generated_attack_data/site5_5.9kw_2016_2019_attack_sa_10pct.csv",
    "./generated_attack_data/site5_327.6kw_2016_2019_attack_sa_5pct.csv",
    "./generated_attack_data/site5_327.6kw_2016_2019_attack_sa_10pct.csv",
]

EVAL_CSVS_SA = [
    "./generated_attack_data/site5_7.0kw_2016_2019_attack_sa_5pct.csv",
    "./generated_attack_data/site5_7.0kw_2016_2019_attack_sa_8pct.csv",
    "./generated_attack_data/site5_7.0kw_2016_2019_attack_sa_10pct.csv",
    "./generated_attack_data/site5_226.8kw_2016_2019_attack_sa_5pct.csv",
    "./generated_attack_data/site5_226.8kw_2016_2019_attack_sa_8pct.csv",
    "./generated_attack_data/site5_226.8kw_2016_2019_attack_sa_10pct.csv",
]

OUT_DIR = "./detector_model_fadre_v3_catboost_depth7"
CACHE_DIR = "./feature_cache_fadre_v3_catboost_depth7"

CATBOOST_MODEL_PATH = "./model_output_catboost_depth_only/cat_depth_7/robust_forecaster_catboost.cbm"
CATBOOST_META_PATH = "./model_output_catboost_depth_only/cat_depth_7/robust_forecaster_catboost_meta.pkl"
LGBM_MODEL_PATH = "./model_output_robust_tweedie/robust_forecaster_lgbm.pkl"

# =============================================================================
# 1. Parameters
# =============================================================================
RANDOM_STATE = 42
VERBOSE_PROGRESS = True
STAGE_TIMER = {}
SCRIPT_START_TIME = time.time()
CACHE_VERSION = "fadre_v3_legacy_lgbm_zone_threshold_fast"  # 피처 생성은 동일하므로 v3.9 캐시 재사용 가능

MIN_GHI_VALID = 200.0
GHI_BINS = {
    "mid":  (300.0, 545.0),
    "high": (545.0, 9999.0),
}

WINDOW_SCALES = [6, 12, 24, 36, 48]
TRAIN_RATIO = 0.80

EPS_RATIO = 0.02
CLIP_PRED_TO_RANGE = True
PRED_MIN = 0.0
PRED_MAX = 1.2

# 실시간성 유지를 위해 residual centering/calibration은 사용하지 않음.
# 잔차는 항상 forecaster의 원시 예측값 y_hat 기준으로 계산한다.

# ═══════════════════════════════════════════════════════
# ★ 여기만 수정하면서 반복 실험 ★
# ═══════════════════════════════════════════════════════

# Round 1: 빈 리스트 (전체 피처)
# Round 2+: importance 보고 쓸모없는 거 추가
DROP_FEATURES = [
    # ── Round 1: 비워둠 (전체 피처로 baseline 측정) ──
]

# zone별 LGB 파라미터 (튜닝 포인트)
ZONE_LGB_PARAMS = {
    "mid":  {"n_estimators": 800, "learning_rate": 0.02, "max_depth": 5,
             "num_leaves": 31, "min_child_samples": 50},
    "high": {"n_estimators": 800, "learning_rate": 0.02, "max_depth": 5,
             "num_leaves": 63, "min_child_samples": 50},
}

# Fixed Threshold 파라미터
# test_ver6 debug grid-search에서 찾은 고정 probability threshold를 최종 판정에 사용한다.
FIXED_PROB_THRESHOLD_BY_ZONE = {
    "mid": 0.45,
    "high": 0.55,
}

# Legacy rolling threshold 파라미터. 디버그/비교용 함수는 남겨두지만 최종 판정에는 쓰지 않는다.
ROLLING_DAYS = 30
THRESHOLD_PCTILE = 99.0
MIN_CLEAN_SAMPLES = 50
FALLBACK_THRESHOLD = 0.25
MID_MIN_PROB = 0.30
HIGH_CONF_MULTIPLIER = 1.5  # legacy 호환용. 실제 판정은 p > dyn_t와 동일하게 동작한다.


# =============================================================================
# 2. Logging
# =============================================================================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _fmt(s):
    s = int(s); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}][+{_fmt(time.time()-SCRIPT_START_TIME)}][{level}] {msg}", flush=True)

def stage_start(n): STAGE_TIMER[n] = time.time(); log(f"START: {n}")
def stage_end(n, x=""):
    e = _fmt(time.time()-STAGE_TIMER.get(n, time.time()))
    log(f"END  : {n} | {e}" + (f" | {x}" if x else ""))


# =============================================================================
# 3. Data utils
# =============================================================================
def load_data(csv_path):
    if not os.path.exists(csv_path): raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path); df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp"] + (["site"] if "site" in df.columns else [])).reset_index(drop=True)
    return df

def normalize_columns(df):
    df = df.copy()
    if "power" not in df.columns and "Active_Power" in df.columns: df["power"] = df["Active_Power"]
    if "Active_Power" not in df.columns and "power" in df.columns: df["Active_Power"] = df["power"]
    if "power_ratio" not in df.columns and {"power","capacity_kw"}.issubset(df.columns):
        df["power_ratio"] = df["power"] / df["capacity_kw"]
    for c, d in [("attack_label",0),("attack_type","clean"),("attack_ratio",0.0),("site","site")]:
        if c not in df.columns: df[c] = d
    return df

def filter_valid_ghi(df): return df[df["ghi"]>=MIN_GHI_VALID].copy().reset_index(drop=True)

def assign_ghi_zone(v):
    # legacy 코드와 맞춤: 200 미만만 ignore.
    # 200~300처럼 명시 bin에 안 걸리는 값은 기존 코드처럼 high로 보낸다.
    if pd.isna(v) or float(v) < MIN_GHI_VALID:
        return "ignore"
    for z, (lo, hi) in GHI_BINS.items():
        if lo <= float(v) < hi:
            return z
    return "high"

def add_date_cols(df):
    df = df.copy(); ts = pd.to_datetime(df["timestamp"])
    df["date"] = ts.dt.date; df["month"] = ts.dt.month; return df

def safe_quantile(v, q, fb):
    v = pd.Series(v).replace([np.inf,-np.inf],np.nan).dropna().values
    return float(np.quantile(v,q)) if len(v)>0 else float(fb)


# =============================================================================
# 4. Forecaster (v3.1 그대로)
# =============================================================================
def add_forecaster_features(df):
    df = normalize_columns(df).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sort_cols = ["site","timestamp"] if "site" in df.columns else ["timestamp"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    ts = df["timestamp"]
    if "sin_hour" not in df.columns:
        h = ts.dt.hour + ts.dt.minute/60.0
        df["sin_hour"]=np.sin(2*np.pi*h/24); df["cos_hour"]=np.cos(2*np.pi*h/24)
    if "sin_doy" not in df.columns:
        d = ts.dt.dayofyear
        df["sin_doy"]=np.sin(2*np.pi*d/365); df["cos_doy"]=np.cos(2*np.pi*d/365)
    grp = "site" if "site" in df.columns else None
    groups = df.groupby(grp, sort=False) if grp else [(None, df)]
    parts = []
    for _, g in groups:
        g = g.copy().sort_values("timestamp")
        for lag in [1,2,3,4,5,6,12,24]:
            c=f"ghi_lag{lag}"
            if c not in g.columns: g[c]=g["ghi"].shift(lag)
        for lag in [1,2,3,6]:
            if f"dghi_{lag}" not in g.columns: g[f"dghi_{lag}"]=g["ghi"]-g["ghi"].shift(lag)
            if f"abs_dghi_{lag}" not in g.columns: g[f"abs_dghi_{lag}"]=g[f"dghi_{lag}"].abs()
            if f"slope_ghi_{lag}" not in g.columns: g[f"slope_ghi_{lag}"]=g[f"dghi_{lag}"]/float(lag)
        if "dghi" not in g.columns: g["dghi"]=g["ghi"]-g["ghi"].shift(1)
        for w in [3,6]:
            if f"ghi_roll_mean_{w}" not in g.columns:
                g[f"ghi_roll_mean_{w}"]=g["ghi"].rolling(w,min_periods=1).mean()
            if f"ghi_roll_std_{w}" not in g.columns:
                g[f"ghi_roll_std_{w}"]=g["ghi"].rolling(w,min_periods=1).std().fillna(0)
            if f"ghi_roll_range_{w}" not in g.columns:
                g[f"ghi_roll_range_{w}"]=g["ghi"].rolling(w,min_periods=1).max()-g["ghi"].rolling(w,min_periods=1).min()
            if f"abs_dghi_1_roll_mean_{w}" not in g.columns:
                g[f"abs_dghi_1_roll_mean_{w}"]=g["abs_dghi_1"].rolling(w,min_periods=1).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(sort_cols).reset_index(drop=True)

class ForecasterWrapper:
    def __init__(self): self.kind=None; self.model=None; self.meta={}
    def load(self):
        # CatBoost forecaster 비교 실험.
        # 같은 detector 설정에서 forecaster만 CatBoost로 바꿔 Tweedie LGBM과 성능을 비교한다.
        if os.path.exists(CATBOOST_MODEL_PATH) and os.path.exists(CATBOOST_META_PATH):
            import joblib
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor()
            self.model.load_model(CATBOOST_MODEL_PATH)
            self.meta = joblib.load(CATBOOST_META_PATH)
            self.kind = "catboost"
            log("CatBoost loaded (forecaster comparison)")
            return self
        raise FileNotFoundError(f"No CatBoost forecaster: {CATBOOST_MODEL_PATH}")
    @property
    def base_features(self):
        for k in ["base_features","features_base","input_features"]:
            if k in self.meta: return list(self.meta[k])
        return ["ghi","temp","sin_hour","cos_hour","sin_doy","cos_doy",
                "ghi_lag1","ghi_lag2","ghi_lag3","ghi_lag4","ghi_lag5","ghi_lag6","ghi_lag12","ghi_lag24",
                "dghi_1","abs_dghi_1","slope_ghi_1","dghi_2","abs_dghi_2","slope_ghi_2",
                "dghi_3","abs_dghi_3","slope_ghi_3","dghi_6","abs_dghi_6","slope_ghi_6",
                "ghi_roll_mean_3","ghi_roll_std_3","ghi_roll_range_3","abs_dghi_1_roll_mean_3",
                "ghi_roll_mean_6","ghi_roll_std_6","ghi_roll_range_6","abs_dghi_1_roll_mean_6","capacity_kw"]
    @property
    def final_features(self):
        for k in ["features","feature_names","final_features"]:
            if k in self.meta: return list(self.meta[k])
        return self.base_features
    @property
    def use_site_onehot(self): return bool(self.meta.get("use_site_onehot",True))
    def make_X(self, df):
        df=add_forecaster_features(df)
        X=df.reindex(columns=self.base_features,fill_value=0.0).copy()
        X=X.replace([np.inf,-np.inf],np.nan).fillna(0.0)
        if self.use_site_onehot and "site" in df.columns:
            X=pd.concat([X,pd.get_dummies(df["site"],prefix="site")],axis=1)
        return X.reindex(columns=self.final_features,fill_value=0.0)
    def predict_raw(self, df):
        p=np.asarray(self.model.predict(self.make_X(df)),dtype=float)
        return np.clip(p,PRED_MIN,PRED_MAX) if CLIP_PRED_TO_RANGE else p
    def add_residuals(self, df):
        """
        Online-compatible residual construction.

        별도의 clean-data calibration/residual-centering을 수행하지 않는다.
        현재 입력에서 forecaster가 산출한 원시 예측값 y_hat만 사용하여
        뺄셈 잔차와 비율 잔차를 계산한다.
        """
        df = normalize_columns(df)
        df = add_forecaster_features(df)
        out = filter_valid_ghi(df)

        y_hat = self.predict_raw(out)
        y_true = out["power_ratio"].values.astype(float)

        out["y_pred"] = y_hat
        out["y_pred_raw"] = y_hat
        out["forecast_bias"] = 0.0

        # raw/subtractive residual: absolute deviation from predicted normal generation ratio
        out["residual_raw"] = y_true - y_hat

        # PR/ratio residual: relative deviation. EPS prevents explosion near zero prediction.
        out["residual_pr"] = np.clip(y_true / (y_hat + EPS_RATIO), 0, 3)



# Fast residual path: add_forecaster_features()를 두 번 호출하지 않도록 ForecasterWrapper.add_residuals를 덮어쓴다.
def _fast_add_residuals(self, df):
    """
    Online-compatible residual construction without clean-data calibration.
    기존 v3.5와 동일한 residual 정의를 유지하되, forecaster feature 생성 중복을 제거한다.
    """
    df = normalize_columns(df)
    df = add_forecaster_features(df)
    out = filter_valid_ghi(df)

    X = out.reindex(columns=self.base_features, fill_value=0.0).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if self.use_site_onehot and "site" in out.columns:
        X = pd.concat([X, pd.get_dummies(out["site"], prefix="site")], axis=1)
    X = X.reindex(columns=self.final_features, fill_value=0.0)

    y_hat = np.asarray(self.model.predict(X), dtype=float)
    if CLIP_PRED_TO_RANGE:
        y_hat = np.clip(y_hat, PRED_MIN, PRED_MAX)
    y_true = out["power_ratio"].values.astype(float)

    out["y_pred"] = y_hat
    out["y_pred_raw"] = y_hat
    out["forecast_bias"] = 0.0
    out["residual_raw"] = y_true - y_hat
    out["residual_pr"] = np.clip(y_true / (y_hat + EPS_RATIO), 0, 3)
    return add_date_cols(out)

ForecasterWrapper.add_residuals = _fast_add_residuals


# =============================================================================
# 5. Feature extraction (v3.1 그대로)
# =============================================================================
if HAS_NUMBA:
    @njit
    def _cusum_fw(w,k):
        cp=cn=mp=mn=0.0
        for v in w: cp=max(0,cp+v-k);cn=max(0,cn-v-k);mp=max(mp,cp);mn=max(mn,cn)
        return max(mp,mn),mp-mn
    @njit
    def _pp(w):
        mx=c=0
        for i in range(len(w)):
            if w[i]>0: c+=1
            else: c=0
            if c>mx: mx=c
        return mx
    @njit
    def _pud(d):
        mx=c=0
        for i in range(len(d)):
            if d[i]>0: c+=1
            else: c=0
            if c>mx: mx=c
        return mx
    @njit
    def _rsi(d):
        ag=al=0.0;n=len(d)
        for i in range(n):
            if d[i]>0: ag+=d[i]
            else: al+=(-d[i])
        ag/=n;al/=n
        return 100*ag/(ag+al) if(ag+al)>0 else 50.0
    @njit
    def _ac(w,m,v):
        if v<1e-10 or len(w)<4: return 0.0
        s=0.0
        for i in range(len(w)-1): s+=(w[i]-m)*(w[i+1]-m)
        return s/((len(w)-1)*v)
else:
    def _cusum_fw(w,k):
        cp=cn=mp=mn=0.0
        for v in w: cp=max(0,cp+v-k);cn=max(0,cn-v-k);mp=max(mp,cp);mn=max(mn,cn)
        return max(mp,mn),mp-mn
    def _pp(w):
        mx=c=0
        for i in range(len(w)):
            if w[i]>0: c+=1;mx=max(mx,c)
            else: c=0
        return mx
    def _pud(d): return _pp(d)
    def _rsi(d):
        ag=np.mean(np.maximum(d,0));al=np.mean(np.abs(np.minimum(d,0)))
        return 100*ag/(ag+al) if(ag+al)>0 else 50.0
    def _ac(w,m,v):
        if v<1e-10 or len(w)<4: return 0.0
        return np.mean((w[:-1]-m)*(w[1:]-m))/v


def _extract_win_feats(raw,pr,ghi,er,ehr,ep,ehp,scales,n):
    kd=0.02; rc={}
    for ws in scales:
        A={k:np.full(n,np.nan) for k in [
            "rm","rs2","rt","rmc","rv","rri","rsk","rku","rac","rcm","rca","rpe","rgc",
            "pd","ps2","pt","pmc","pv","pri","psk","pku","pac","pcm","pca","ppu","pgc",
            "gam","gax","gas"]}
        for i in range(ws-1,n):
            s=i-ws+1; wr=raw[s:i+1]; wp=pr[s:i+1]; wg=ghi[s:i+1]; gs=np.std(wg)
            rm=np.mean(wr);rs=np.std(wr)
            A["rm"][i]=rm; A["rs2"][i]=rm/(rs+1e-6); A["rt"][i]=(wr[-1]-wr[0])/ws
            A["rmc"][i]=ehr[ws][i]-er[ws][i]
            A["rv"][i]=rs/(gs/1000) if gs>1e-6 else rs*100
            dr=np.diff(wr); A["rri"][i]=_rsi(dr)
            if len(wr)>=4:
                sk=scipy_skew(wr);A["rsk"][i]=0 if np.isnan(sk) else sk
                kt=kurtosis(wr);A["rku"][i]=0 if np.isnan(kt) else kt
            else: A["rsk"][i]=0;A["rku"][i]=0
            A["rac"][i]=_ac(wr,rm,np.var(wr))
            cm,ca=_cusum_fw(wr,kd);A["rcm"][i]=cm;A["rca"][i]=ca
            A["rpe"][i]=_pp(wr)
            if len(wr)>=3 and gs>1e-6 and rs>1e-6:
                c=np.corrcoef(np.diff(wg.astype(float)),dr)[0,1];A["rgc"][i]=0 if np.isnan(c) else c
            else: A["rgc"][i]=0

            wd=wp-1.0;ps=np.std(wp)
            A["pd"][i]=np.mean(wd);A["ps2"][i]=np.mean(wd)/(ps+1e-6);A["pt"][i]=(wp[-1]-wp[0])/ws
            A["pmc"][i]=ehp[ws][i]-ep[ws][i]
            A["pv"][i]=ps/(gs/1000) if gs>1e-6 else ps*100
            dp=np.diff(wp);A["pri"][i]=_rsi(dp)
            if len(wd)>=4:
                sk=scipy_skew(wd);A["psk"][i]=0 if np.isnan(sk) else sk
                kt=kurtosis(wd);A["pku"][i]=0 if np.isnan(kt) else kt
            else: A["psk"][i]=0;A["pku"][i]=0
            A["pac"][i]=_ac(wd,np.mean(wd),np.var(wd))
            cm,ca=_cusum_fw(wd,kd);A["pcm"][i]=cm;A["pca"][i]=ca
            A["ppu"][i]=_pud(dp)
            if len(wp)>=3 and gs>1e-6 and ps>1e-6:
                c=np.corrcoef(np.diff(wg.astype(float)),dp)[0,1];A["pgc"][i]=0 if np.isnan(c) else c
            else: A["pgc"][i]=0

            if len(wg)>=3:
                ga=np.abs(np.diff(wg.astype(float),n=2))
                A["gam"][i]=np.mean(ga);A["gax"][i]=np.max(ga);A["gas"][i]=np.std(ga)
            else: A["gam"][i]=0;A["gax"][i]=0;A["gas"][i]=0

        nm={"rm":"F_raw_mean","rs2":"F_raw_snr","rt":"F_raw_trend","rmc":"F_raw_macd",
            "rv":"F_raw_vol_ratio","rri":"F_raw_rsi","rsk":"F_raw_skew","rku":"F_raw_kurtosis",
            "rac":"F_raw_autocorr","rcm":"F_raw_cusum_max","rca":"F_raw_cusum_asym",
            "rpe":"F_raw_persist","rgc":"F_raw_ghi_coh",
            "pd":"F_pr_dev_mean","ps2":"F_pr_snr","pt":"F_pr_trend","pmc":"F_pr_macd",
            "pv":"F_pr_vol_ratio","pri":"F_pr_rsi","psk":"F_pr_skew","pku":"F_pr_kurtosis",
            "pac":"F_pr_autocorr","pcm":"F_pr_cusum_max","pca":"F_pr_cusum_asym",
            "ppu":"F_pr_persist_up","pgc":"F_pr_ghi_coh",
            "gam":"F_ghi_accel_mean","gax":"F_ghi_accel_max","gas":"F_ghi_accel_std"}
        for k,a in A.items(): rc[f"{nm[k]}_{ws}"]=a
    return rc


def extract_ml_features(df, scales, tag=""):
    df=df.sort_values("timestamp").reset_index(drop=True)
    n=len(df)
    raw=df["residual_raw"].astype(float).values; pr=df["residual_pr"].astype(float).values
    ghi=df["ghi"].astype(float).values
    er={w:df["residual_raw"].ewm(span=w,adjust=False).mean().values for w in scales}
    ehr={w:df["residual_raw"].ewm(span=max(1,w//2),adjust=False).mean().values for w in scales}
    ep={w:df["residual_pr"].ewm(span=w,adjust=False).mean().values for w in scales}
    ehp={w:df["residual_pr"].ewm(span=max(1,w//2),adjust=False).mean().values for w in scales}

    res=pd.DataFrame({
        "timestamp":df["timestamp"].values,
        "date":df["date"].values if "date" in df.columns else pd.to_datetime(df["timestamp"]).dt.date,
        "ghi":ghi,"temp":df["temp"].values if "temp" in df.columns else np.nan,
        "ghi_zone":df["ghi"].apply(assign_ghi_zone).values,
        "is_attack":(df["attack_label"].values>0.2).astype(int) if "attack_label" in df.columns else 0,
        "attack_type":df["attack_type"].values if "attack_type" in df.columns else "clean",
        "attack_ratio":df["attack_ratio"].values.astype(float) if "attack_ratio" in df.columns else 0.0,
    })
    gr6=np.zeros(n)
    for i in range(1,n): w=ghi[max(0,i-5):i+1]; gr6[i]=np.max(w)-np.min(w)
    res["F_ghi_range_6"]=gr6
    ga=np.zeros(n)
    for i in range(2,n): ga[i]=abs(ghi[i]-2*ghi[i-1]+ghi[i-2])
    res["F_ghi_accel"]=ga
    wc=_extract_win_feats(raw,pr,ghi,er,ehr,ep,ehp,scales,n)
    for k,v in wc.items(): res[k]=v
    before=len(res); res=res.dropna().reset_index(drop=True)
    if VERBOSE_PROGRESS: log(f"features: {tag} before={before:,} after={len(res):,}")
    return res




# =============================================================================
# 5-B. Fast feature extraction override
# =============================================================================
# 기존 _extract_win_feats()는 Python 이중 루프 + scipy skew/kurtosis + np.corrcoef 때문에 느리다.
# 아래 fast 버전은 같은 계열 피처를 numba 단일 루프로 계산한다.
_extract_win_feats_slow = _extract_win_feats
extract_ml_features_slow = extract_ml_features


def _feature_names_for_scales(scales):
    base_names = [
        "F_raw_mean", "F_raw_snr", "F_raw_trend", "F_raw_macd", "F_raw_vol_ratio",
        "F_raw_rsi", "F_raw_skew", "F_raw_kurtosis", "F_raw_autocorr",
        "F_raw_cusum_max", "F_raw_cusum_asym", "F_raw_persist", "F_raw_ghi_coh",
        "F_pr_dev_mean", "F_pr_snr", "F_pr_trend", "F_pr_macd", "F_pr_vol_ratio",
        "F_pr_rsi", "F_pr_skew", "F_pr_kurtosis", "F_pr_autocorr",
        "F_pr_cusum_max", "F_pr_cusum_asym", "F_pr_persist_up", "F_pr_ghi_coh",
        "F_ghi_accel_mean", "F_ghi_accel_max", "F_ghi_accel_std",
    ]
    names = []
    for ws in scales:
        for b in base_names:
            names.append(f"{b}_{int(ws)}")
    return names


if HAS_NUMBA:
    @njit(cache=True)
    def _rolling_range6_numba(ghi):
        n = len(ghi)
        out = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            s = i - 5
            if s < 0:
                s = 0
            mn = ghi[s]
            mx = ghi[s]
            for j in range(s + 1, i + 1):
                v = ghi[j]
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
            out[i] = mx - mn
        return out

    @njit(cache=True)
    def _ghi_accel_numba(ghi):
        n = len(ghi)
        out = np.zeros(n, dtype=np.float64)
        for i in range(2, n):
            out[i] = abs(ghi[i] - 2.0 * ghi[i - 1] + ghi[i - 2])
        return out

    @njit(cache=True)
    def _extract_win_feats_fast_numba(raw, pr, ghi, er, ehr, ep, ehp, scales):
        n = len(raw)
        ns = len(scales)
        nf_per = 29
        M = np.empty((n, ns * nf_per), dtype=np.float64)
        for i in range(n):
            for j in range(ns * nf_per):
                M[i, j] = np.nan

        kd = 0.02

        for si in range(ns):
            ws = int(scales[si])
            base = si * nf_per
            for i in range(ws - 1, n):
                s = i - ws + 1

                # ---------- basic means/std ----------
                sum_r = 0.0
                sum_p = 0.0
                sum_g = 0.0
                for j in range(s, i + 1):
                    sum_r += raw[j]
                    sum_p += pr[j]
                    sum_g += ghi[j]
                rm = sum_r / ws
                pm = sum_p / ws
                gm = sum_g / ws

                var_r = 0.0
                var_p = 0.0
                var_g = 0.0
                m3_r = 0.0
                m4_r = 0.0
                m3_p = 0.0
                m4_p = 0.0
                for j in range(s, i + 1):
                    dr0 = raw[j] - rm
                    dp0 = pr[j] - pm
                    dg0 = ghi[j] - gm
                    dr2 = dr0 * dr0
                    dp2 = dp0 * dp0
                    var_r += dr2
                    var_p += dp2
                    var_g += dg0 * dg0
                    m3_r += dr2 * dr0
                    m4_r += dr2 * dr2
                    m3_p += dp2 * dp0
                    m4_p += dp2 * dp2
                var_r /= ws
                var_p /= ws
                var_g /= ws
                rs = np.sqrt(var_r)
                ps = np.sqrt(var_p)
                gs = np.sqrt(var_g)

                # raw skew/kurtosis, scipy default에 가까운 biased moment 방식
                if ws >= 4 and var_r > 1e-12:
                    raw_skew = (m3_r / ws) / (rs * rs * rs)
                    raw_kurt = (m4_r / ws) / (var_r * var_r) - 3.0
                else:
                    raw_skew = 0.0
                    raw_kurt = 0.0
                if ws >= 4 and var_p > 1e-12:
                    pr_skew = (m3_p / ws) / (ps * ps * ps)
                    pr_kurt = (m4_p / ws) / (var_p * var_p) - 3.0
                else:
                    pr_skew = 0.0
                    pr_kurt = 0.0

                # ---------- raw diff stats ----------
                gain_r = 0.0
                loss_r = 0.0
                gain_p = 0.0
                loss_p = 0.0
                max_persist_raw = 0
                cur_persist_raw = 0
                max_persist_up = 0
                cur_persist_up = 0

                sum_dg = 0.0
                sum_dr = 0.0
                sum_dp = 0.0
                nd = ws - 1
                for j in range(s + 1, i + 1):
                    dgr = ghi[j] - ghi[j - 1]
                    drr = raw[j] - raw[j - 1]
                    dpp = pr[j] - pr[j - 1]
                    sum_dg += dgr
                    sum_dr += drr
                    sum_dp += dpp
                    if drr > 0.0:
                        gain_r += drr
                    else:
                        loss_r += -drr
                    if dpp > 0.0:
                        gain_p += dpp
                        cur_persist_up += 1
                    else:
                        loss_p += -dpp
                        cur_persist_up = 0
                    if cur_persist_up > max_persist_up:
                        max_persist_up = cur_persist_up

                raw_rsi = 100.0 * gain_r / (gain_r + loss_r) if (gain_r + loss_r) > 0.0 else 50.0
                pr_rsi = 100.0 * gain_p / (gain_p + loss_p) if (gain_p + loss_p) > 0.0 else 50.0

                for j in range(s, i + 1):
                    if raw[j] > 0.0:
                        cur_persist_raw += 1
                    else:
                        cur_persist_raw = 0
                    if cur_persist_raw > max_persist_raw:
                        max_persist_raw = cur_persist_raw

                # ---------- autocorr ----------
                raw_ac = 0.0
                pr_ac = 0.0
                if ws >= 4 and var_r > 1e-10:
                    acc = 0.0
                    for j in range(s, i):
                        acc += (raw[j] - rm) * (raw[j + 1] - rm)
                    raw_ac = acc / ((ws - 1) * var_r)
                if ws >= 4 and var_p > 1e-10:
                    acc = 0.0
                    for j in range(s, i):
                        acc += (pr[j] - pm) * (pr[j + 1] - pm)
                    pr_ac = acc / ((ws - 1) * var_p)

                # ---------- CUSUM ----------
                cp = 0.0
                cn = 0.0
                mp = 0.0
                mn = 0.0
                for j in range(s, i + 1):
                    v = raw[j]
                    cp = max(0.0, cp + v - kd)
                    cn = max(0.0, cn - v - kd)
                    if cp > mp:
                        mp = cp
                    if cn > mn:
                        mn = cn
                raw_cusum_max = mp if mp > mn else mn
                raw_cusum_asym = mp - mn

                cp = 0.0
                cn = 0.0
                mp = 0.0
                mn = 0.0
                for j in range(s, i + 1):
                    v = pr[j] - 1.0
                    cp = max(0.0, cp + v - kd)
                    cn = max(0.0, cn - v - kd)
                    if cp > mp:
                        mp = cp
                    if cn > mn:
                        mn = cn
                pr_cusum_max = mp if mp > mn else mn
                pr_cusum_asym = mp - mn

                # ---------- correlation with GHI diff ----------
                raw_ghi_coh = 0.0
                pr_ghi_coh = 0.0
                if ws >= 3 and gs > 1e-6 and rs > 1e-6:
                    mdg = sum_dg / nd
                    mdr = sum_dr / nd
                    cov = 0.0
                    vdg = 0.0
                    vdr = 0.0
                    for j in range(s + 1, i + 1):
                        a = (ghi[j] - ghi[j - 1]) - mdg
                        b = (raw[j] - raw[j - 1]) - mdr
                        cov += a * b
                        vdg += a * a
                        vdr += b * b
                    den = np.sqrt(vdg * vdr)
                    raw_ghi_coh = cov / den if den > 1e-12 else 0.0
                if ws >= 3 and gs > 1e-6 and ps > 1e-6:
                    mdg = sum_dg / nd
                    mdp = sum_dp / nd
                    cov = 0.0
                    vdg = 0.0
                    vdp = 0.0
                    for j in range(s + 1, i + 1):
                        a = (ghi[j] - ghi[j - 1]) - mdg
                        b = (pr[j] - pr[j - 1]) - mdp
                        cov += a * b
                        vdg += a * a
                        vdp += b * b
                    den = np.sqrt(vdg * vdp)
                    pr_ghi_coh = cov / den if den > 1e-12 else 0.0

                # ---------- GHI acceleration in window ----------
                gam = 0.0
                gax = 0.0
                gas = 0.0
                if ws >= 3:
                    cnt = ws - 2
                    sacc = 0.0
                    sacc2 = 0.0
                    maxacc = 0.0
                    for j in range(s + 2, i + 1):
                        a = abs(ghi[j] - 2.0 * ghi[j - 1] + ghi[j - 2])
                        sacc += a
                        sacc2 += a * a
                        if a > maxacc:
                            maxacc = a
                    gam = sacc / cnt
                    vv = sacc2 / cnt - gam * gam
                    if vv < 0.0:
                        vv = 0.0
                    gas = np.sqrt(vv)
                    gax = maxacc

                # ---------- fill matrix ----------
                M[i, base + 0] = rm
                M[i, base + 1] = rm / (rs + 1e-6)
                M[i, base + 2] = (raw[i] - raw[s]) / ws
                M[i, base + 3] = ehr[si, i] - er[si, i]
                M[i, base + 4] = rs / (gs / 1000.0) if gs > 1e-6 else rs * 100.0
                M[i, base + 5] = raw_rsi
                M[i, base + 6] = raw_skew
                M[i, base + 7] = raw_kurt
                M[i, base + 8] = raw_ac
                M[i, base + 9] = raw_cusum_max
                M[i, base + 10] = raw_cusum_asym
                M[i, base + 11] = max_persist_raw
                M[i, base + 12] = raw_ghi_coh

                dev_mean = pm - 1.0
                M[i, base + 13] = dev_mean
                M[i, base + 14] = dev_mean / (ps + 1e-6)
                M[i, base + 15] = (pr[i] - pr[s]) / ws
                M[i, base + 16] = ehp[si, i] - ep[si, i]
                M[i, base + 17] = ps / (gs / 1000.0) if gs > 1e-6 else ps * 100.0
                M[i, base + 18] = pr_rsi
                M[i, base + 19] = pr_skew
                M[i, base + 20] = pr_kurt
                M[i, base + 21] = pr_ac
                M[i, base + 22] = pr_cusum_max
                M[i, base + 23] = pr_cusum_asym
                M[i, base + 24] = max_persist_up
                M[i, base + 25] = pr_ghi_coh

                M[i, base + 26] = gam
                M[i, base + 27] = gax
                M[i, base + 28] = gas
        
        return M


def _assign_ghi_zone_vectorized(ghi):
    z = np.full(len(ghi), "ignore", dtype=object)
    for name, (lo, hi) in GHI_BINS.items():
        m = (ghi >= lo) & (ghi < hi)
        z[m] = name
    return z


def extract_ml_features(df, scales, tag=""):
    if not HAS_NUMBA:
        log("numba not available: using slow Python feature extraction", "WARN")
        return extract_ml_features_slow(df, scales, tag=tag)

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    raw = df["residual_raw"].astype(float).values
    pr = df["residual_pr"].astype(float).values
    ghi = df["ghi"].astype(float).values
    scales_arr = np.asarray(scales, dtype=np.int64)

    # ewm은 pandas가 빠르므로 그대로 사용하고, window 통계 루프만 numba로 넘긴다.
    er = np.vstack([df["residual_raw"].ewm(span=int(w), adjust=False).mean().values for w in scales_arr])
    ehr = np.vstack([df["residual_raw"].ewm(span=max(1, int(w)//2), adjust=False).mean().values for w in scales_arr])
    ep = np.vstack([df["residual_pr"].ewm(span=int(w), adjust=False).mean().values for w in scales_arr])
    ehp = np.vstack([df["residual_pr"].ewm(span=max(1, int(w)//2), adjust=False).mean().values for w in scales_arr])

    names = _feature_names_for_scales(scales_arr)
    M = _extract_win_feats_fast_numba(raw, pr, ghi, er, ehr, ep, ehp, scales_arr)

    res = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "date": df["date"].values if "date" in df.columns else pd.to_datetime(df["timestamp"]).dt.date,
        "ghi": ghi,
        "temp": df["temp"].values if "temp" in df.columns else np.nan,
        "ghi_zone": _assign_ghi_zone_vectorized(ghi),
        "is_attack": (df["attack_label"].values > 0.2).astype(int) if "attack_label" in df.columns else 0,
        "attack_type": df["attack_type"].values if "attack_type" in df.columns else "clean",
        "attack_ratio": df["attack_ratio"].values.astype(float) if "attack_ratio" in df.columns else 0.0,
    })

    # 기존과 같은 단일 GHI 피처 2개
    res["F_ghi_range_6"] = _rolling_range6_numba(ghi)
    res["F_ghi_accel"] = _ghi_accel_numba(ghi)

    # 한 번에 concat해서 DataFrame fragmentation 방지
    feat_df = pd.DataFrame(M, columns=names)
    res = pd.concat([res, feat_df], axis=1)

    before = len(res)
    res = res.dropna().reset_index(drop=True)
    if VERBOSE_PROGRESS:
        log(f"features-fast: {tag} before={before:,} after={len(res):,}")
    return res

# =============================================================================
# 6. Cache
# =============================================================================
def cache_path_for(tag, src, sfx):
    mt=[f"{p}:{os.path.getmtime(p):.0f}" for p in src if os.path.exists(p)]
    k="|".join([CACHE_VERSION,tag,sfx,str(MIN_GHI_VALID),str(GHI_BINS),str(WINDOW_SCALES)]+mt)
    return os.path.join(CACHE_DIR,f"{tag}_{sfx}_{hashlib.md5(k.encode()).hexdigest()[:12]}.pkl")

def load_residual_seq(csv, fc, tag):
    ensure_dir(CACHE_DIR); cp=cache_path_for(tag,[csv,CATBOOST_MODEL_PATH,LGBM_MODEL_PATH],"resid")
    if os.path.exists(cp):
        with open(cp,"rb") as f: return pickle.load(f)
    df=load_data(csv); df=fc.add_residuals(df)
    with open(cp,"wb") as f: pickle.dump(df,f)
    return df

def extract_feat_cached(df, tag):
    ensure_dir(CACHE_DIR); cp=cache_path_for(tag,[],"features_v3")
    if os.path.exists(cp):
        with open(cp,"rb") as f: return pickle.load(f)
    w=df.copy(); w["ghi_zone"]=w["ghi"].apply(assign_ghi_zone)
    feat=extract_ml_features(w,WINDOW_SCALES,tag=tag)
    with open(cp,"wb") as f: pickle.dump(feat,f)
    return feat


# =============================================================================
# 7. Rolling Dynamic Threshold
# =============================================================================
def _rolling_slope_values(arr, window):
    """단순 rolling slope. 입력은 1D array, 출력은 같은 길이의 slope array."""
    arr = np.asarray(arr, dtype=float)
    out = np.full(len(arr), 0.0, dtype=float)
    if len(arr) < 2:
        return out
    for i in range(len(arr)):
        s = max(0, i - window + 1)
        y = arr[s:i+1]
        if len(y) < 3:
            out[i] = 0.0
            continue
        x = np.arange(len(y), dtype=float)
        xv = x - x.mean()
        yv = y - y.mean()
        den = np.sum(xv * xv)
        out[i] = float(np.sum(xv * yv) / den) if den > 1e-12 else 0.0
    return out


def compute_rolling_threshold(df, prob_col="attack_prob"):
    """
    Legacy dynamic threshold decision.

    기존 잘 되던 코드의 판정부를 최대한 그대로 복구한 버전이다.
    - zone별로 최근 ROLLING_DAYS 동안 '정상으로 판단된 확률'만 buffer에 저장
    - buffer의 THRESHOLD_PCTILE percentile을 dynamic threshold로 사용
    - p_t > dynamic_threshold 이면 바로 attack
    - smoothing, long-track, slope guard 없음
    - mid 구간은 p < MID_MIN_PROB이면 detection을 제거

    주의: HIGH_CONF_MULTIPLIER는 legacy 코드에 있었지만,
    기존 if/elif 구조상 최종적으로 p > dyn_t와 동일하게 작동했다.
    따라서 여기서는 명시적으로 p > dyn_t만 사용한다.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    thresholds = np.full(n, FALLBACK_THRESHOLD, dtype=float)
    pred = np.zeros(n, dtype=int)

    dates = df["date"].values
    probs = df[prob_col].astype(float).values
    zones = df["ghi_zone"].values
    unique_dates = sorted(df["date"].unique())

    date_to_idx = {}
    for i, d in enumerate(dates):
        date_to_idx.setdefault(d, []).append(i)

    clean_prob_buffers = {z: {} for z in GHI_BINS}

    for day_idx, current_date in enumerate(unique_dates):
        current_indices = date_to_idx.get(current_date, [])
        past_dates = unique_dates[max(0, day_idx - ROLLING_DAYS):day_idx]

        for idx in current_indices:
            zone = zones[idx]
            p = probs[idx]

            if zone not in GHI_BINS:
                continue

            past_clean = []
            for pd_date in past_dates:
                if pd_date in clean_prob_buffers[zone]:
                    past_clean.extend(clean_prob_buffers[zone][pd_date])

            if len(past_clean) >= MIN_CLEAN_SAMPLES:
                dyn_t = max(float(np.percentile(past_clean, THRESHOLD_PCTILE)), FALLBACK_THRESHOLD)
            elif len(past_clean) > 10:
                raw_p = float(np.percentile(past_clean, THRESHOLD_PCTILE))
                blend = len(past_clean) / MIN_CLEAN_SAMPLES
                dyn_t = blend * raw_p + (1.0 - blend) * FALLBACK_THRESHOLD
                dyn_t = max(float(dyn_t), FALLBACK_THRESHOLD)
            else:
                dyn_t = FALLBACK_THRESHOLD

            thresholds[idx] = dyn_t

            if p > dyn_t:
                pred[idx] = 1
            else:
                pred[idx] = 0
                clean_prob_buffers[zone].setdefault(current_date, []).append(float(p))

    df["dynamic_threshold"] = thresholds
    df["raw_pred"] = pred.copy()
    df["short_raw_pred"] = pred.copy()

    # 기존 mid gate: mid zone에서 확률이 너무 낮은 threshold crossing 제거
    df.loc[
        (df["ghi_zone"] == "mid") &
        (df["short_raw_pred"] == 1) &
        (df[prob_col] < MID_MIN_PROB),
        "short_raw_pred"
    ] = 0

    df["short_pred"] = df["short_raw_pred"].astype(int)
    df["long_pred"] = 0
    df["final_pred"] = df["short_pred"].astype(int)
    return df



# =============================================================================
# 8. Evaluation
# =============================================================================
def pa(y_true, y_pred):
    """Point Adjustment 평가: 실제 공격 구간 안에서 하나라도 맞추면 해당 구간 전체를 탐지로 간주."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    ya = yp.copy()

    in_attack = False
    start = 0
    for i in range(len(yt)):
        if yt[i] == 1 and not in_attack:
            in_attack = True
            start = i
        elif yt[i] == 0 and in_attack:
            in_attack = False
            if np.any(yp[start:i] == 1):
                ya[start:i] = 1
    if in_attack and np.any(yp[start:] == 1):
        ya[start:] = 1
    return ya


def evaluate(df, pred_col, label=""):
    """PA 기반 point metric + raw prediction 기반 day metric."""
    if len(df) == 0:
        return {
            "label": label,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "fpr": 0.0,
            "day_recall": 0.0, "day_fpr": 0.0,
            "n_atk_days": 0, "det_atk_days": 0,
            "n_cln_days": 0, "fa_days": 0,
        }

    yt = df["is_attack"].values.astype(int)
    yp = df[pred_col].values.astype(int)
    ya = pa(yt, yp)

    cm = confusion_matrix(yt, ya, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    daily = df.groupby("date").agg(has_attack=("is_attack", "max")).reset_index()
    atk_days = set(daily[daily["has_attack"] == 1]["date"].values)
    cln_days = set(daily[daily["has_attack"] == 0]["date"].values)
    det_days = set(df[df[pred_col].astype(int) == 1]["date"].values)

    day_recall = len(det_days & atk_days) / len(atk_days) if atk_days else 0.0
    day_fpr = len(det_days & cln_days) / len(cln_days) if cln_days else 0.0

    return {
        "label": label,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1), "fpr": float(fpr),
        "day_recall": float(day_recall), "day_fpr": float(day_fpr),
        "n_atk_days": int(len(atk_days)), "det_atk_days": int(len(det_days & atk_days)),
        "n_cln_days": int(len(cln_days)), "fa_days": int(len(det_days & cln_days)),
    }


def pm(title, m):
    print(f"""
  ┌─────────────────────────────────────┐
  │   {title:<35s} │
  ├─────────────────────────────────────┤
  │  PA Precision : {m['precision']:.4f}               │
  │  PA Recall    : {m['recall']:.4f}               │
  │  F1-Score     : {m['f1']:.4f}               │
  │  FP 건수      : {m['fp']:>5d}건               │
  │  FPR          : {m['fpr']*100:.3f}%               │
  ├─────────────────────────────────────┤
  │  Day Recall   : {m['day_recall']:.4f} ({m['det_atk_days']}/{m['n_atk_days']})       │
  │  Day FPR      : {m['day_fpr']:.4f} ({m['fa_days']}/{m['n_cln_days']})          │
  └─────────────────────────────────────┘""")

def build_train_features(fc):
    pieces=[]
    for p in TRAIN_ATTACK_CSVS:
        tag=Path(p).stem; r=load_residual_seq(p,fc,tag); f=extract_feat_cached(r,tag)
        pieces.append(f); log(f"  {tag}: {len(f):,} (atk={int(f['is_attack'].sum()):,})")
    df=pd.concat(pieces,ignore_index=True)
    return df[df["ghi_zone"]!="ignore"].reset_index(drop=True)

def select_zone_features(df):
    """
    기존에 잘 되던 구조에 맞춰 zone별 feature stream을 분리한다.

    - high zone: raw residual features + GHI dynamics
      높은 일사량에서는 발전량 절대 편차가 안정적으로 관찰되므로 raw residual 계열을 사용한다.

    - mid zone: PR/ratio residual features + GHI dynamics
      중간 일사량에서는 예측 발전량 크기에 따른 상대 변화가 중요하므로 PR 계열을 사용한다.

    DROP_FEATURES는 공통 제거 리스트로만 적용한다. zone-wise pruning은 여기서는 의도적으로 하지 않는다.
    """
    drop = [c for c in DROP_FEATURES if c in df.columns]
    tmp = df.drop(columns=drop, errors="ignore")

    ghi_feats = sorted([c for c in tmp.columns if c.startswith("F_ghi_")])
    raw_feats = sorted([c for c in tmp.columns if c.startswith("F_raw_")])
    pr_feats  = sorted([c for c in tmp.columns if c.startswith("F_pr_")])

    high_feats = sorted(set(raw_feats + ghi_feats))
    mid_feats  = sorted(set(pr_feats + ghi_feats))

    return {"high": high_feats, "mid": mid_feats}, drop

def train_zone_models(dtr,dva,zfc):
    models={}; vp=[]
    for z in GHI_BINS:
        zt=dtr[dtr["ghi_zone"]==z]; zv=dva[dva["ghi_zone"]==z].copy()
        feat=zfc[z]
        if len(zt)==0 or len(zv)==0: continue
        Xt=zt[feat].replace([np.inf,-np.inf],np.nan).fillna(0)
        yt=zt["is_attack"].astype(int)
        Xv=zv[feat].replace([np.inf,-np.inf],np.nan).fillna(0)
        yv=zv["is_attack"].astype(int)
        pos=int(yt.sum());neg=len(yt)-pos
        spw=min((neg/max(pos,1))*2.5,5.0) if pos>0 else 1.0
        p=ZONE_LGB_PARAMS[z]
        clf=lgb.LGBMClassifier(n_estimators=p["n_estimators"],learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],num_leaves=p["num_leaves"],min_child_samples=p["min_child_samples"],
            scale_pos_weight=spw,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,
            random_state=RANDOM_STATE,verbose=-1,n_jobs=-1)
        clf.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(0)])
        zv["attack_prob"]=clf.predict_proba(Xv)[:,1]
        vp.append(zv); models[z]=clf
        log(f"  [{z}] pos={pos} spw={spw:.1f} feat={len(feat)} best={getattr(clf,'best_iteration_','?')}")
    return models, pd.concat(vp,ignore_index=True).sort_values("timestamp").reset_index(drop=True)

def predict_prob(df,models,zfc):
    out=df.copy().reset_index(drop=True); out["attack_prob"]=0.0
    for z,clf in models.items():
        m=out["ghi_zone"]==z
        if not m.any(): continue
        X=out.loc[m,zfc[z]].replace([np.inf,-np.inf],np.nan).fillna(0)
        out.loc[m,"attack_prob"]=clf.predict_proba(X)[:,1]
    return out




# =============================================================================
# 9.5 Debug / threshold tuning tools
# =============================================================================
# Fixed-threshold final run: keep debug helpers available, but do not print sweeps/grid-search.
DEBUG_TUNING = False
DEBUG_ONLY_TAG_CONTAINS = "10pct"  # ""로 바꾸면 모든 평가 CSV 디버그 출력
DEBUG_TOP_K_GRID = 40

# 중요: 아래 sweep/grid는 최종 논문용 threshold 선택이 아니라 원인 분석용이다.
# 최종 threshold는 validation에서 고른 뒤 test에 고정 적용해야 test leakage를 피할 수 있다.


def _evaluate_with_optional_pa(df, pred_col, label="", use_pa=True):
    if use_pa:
        return evaluate(df, pred_col, label)

    if len(df) == 0:
        return {"label": label, "tp":0,"fp":0,"fn":0,"tn":0,
                "precision":0.0,"recall":0.0,"f1":0.0,"fpr":0.0,
                "day_recall":0.0,"day_fpr":0.0,
                "n_atk_days":0,"det_atk_days":0,"n_cln_days":0,"fa_days":0}

    yt = df["is_attack"].values.astype(int)
    yp = df[pred_col].values.astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    fpr = fp/(fp+tn) if (fp+tn) else 0.0

    daily = df.groupby("date").agg(has_attack=("is_attack", "max")).reset_index()
    atk_days = set(daily[daily["has_attack"] == 1]["date"].values)
    cln_days = set(daily[daily["has_attack"] == 0]["date"].values)
    det_days = set(df[df[pred_col].astype(int) == 1]["date"].values)
    day_recall = len(det_days & atk_days) / len(atk_days) if atk_days else 0.0
    day_fpr = len(det_days & cln_days) / len(cln_days) if cln_days else 0.0

    return {"label": label, "tp":int(tp), "fp":int(fp), "fn":int(fn), "tn":int(tn),
            "precision":float(precision), "recall":float(recall), "f1":float(f1), "fpr":float(fpr),
            "day_recall":float(day_recall), "day_fpr":float(day_fpr),
            "n_atk_days":int(len(atk_days)), "det_atk_days":int(len(det_days & atk_days)),
            "n_cln_days":int(len(cln_days)), "fa_days":int(len(det_days & cln_days))}


def debug_score_distribution(df, prob_col="attack_prob", pred_col=None, name=""):
    print("\n" + "="*90)
    print(f"[DEBUG-1] Score distribution | {name}")
    print("="*90)
    if prob_col not in df.columns:
        print(f"  ! Missing prob_col={prob_col}")
        return

    for z in ["mid", "high"]:
        sub = df[df["ghi_zone"] == z].copy()
        if len(sub) == 0:
            continue
        print(f"\n[{z.upper()}] n={len(sub):,}")
        for lab, lab_name in [(0, "clean"), (1, "attack")]:
            s = sub[sub["is_attack"].astype(int) == lab][prob_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) == 0:
                print(f"  {lab_name:<6s} n=0")
                continue
            qs = s.quantile([0.10, 0.50, 0.90, 0.95, 0.99, 0.995, 0.999])
            print(
                f"  {lab_name:<6s} n={len(s):,} "
                f"p10={qs.loc[0.10]:.4f} p50={qs.loc[0.50]:.4f} "
                f"p90={qs.loc[0.90]:.4f} p95={qs.loc[0.95]:.4f} "
                f"p99={qs.loc[0.99]:.4f} p99.5={qs.loc[0.995]:.4f} p99.9={qs.loc[0.999]:.4f}"
            )

        if pred_col is not None and pred_col in sub.columns:
            fp = sub[(sub["is_attack"].astype(int) == 0) & (sub[pred_col].astype(int) == 1)]
            tp = sub[(sub["is_attack"].astype(int) == 1) & (sub[pred_col].astype(int) == 1)]
            print(f"  pred summary: clean FP={len(fp):,}, attack TP={len(tp):,}")
            if len(fp) > 0:
                qfp = fp[prob_col].quantile([0.50, 0.90, 0.99])
                print(f"  FP prob: p50={qfp.loc[0.50]:.4f} p90={qfp.loc[0.90]:.4f} p99={qfp.loc[0.99]:.4f}")


def _apply_zone_fixed_threshold(df, prob_col, mid_th, high_th):
    out = df.copy().reset_index(drop=True)
    out["fixed_pred"] = 0
    mid_mask = out["ghi_zone"] == "mid"
    high_mask = out["ghi_zone"] == "high"
    out.loc[mid_mask, "fixed_pred"] = (out.loc[mid_mask, prob_col] >= mid_th).astype(int)
    out.loc[high_mask, "fixed_pred"] = (out.loc[high_mask, prob_col] >= high_th).astype(int)
    return out


def apply_final_fixed_threshold(df, prob_col="attack_prob"):
    out = _apply_zone_fixed_threshold(
        df,
        prob_col=prob_col,
        mid_th=FIXED_PROB_THRESHOLD_BY_ZONE["mid"],
        high_th=FIXED_PROB_THRESHOLD_BY_ZONE["high"],
    )
    out["raw_pred"] = out["fixed_pred"].astype(int)
    out["short_raw_pred"] = out["fixed_pred"].astype(int)
    out["short_pred"] = out["fixed_pred"].astype(int)
    out["long_pred"] = 0
    out["final_pred"] = out["fixed_pred"].astype(int)
    return out


def _apply_zone_smoothing(df, pred_col="fixed_pred", out_col="smooth_pred", windows=None, min_hits=None):
    # zone별로 rolling hit-count smoothing 적용
    # 예: high는 최근 5개 중 3~4개 이상이어야 유지.
    if windows is None:
        windows = {"mid": 5, "high": 5}
    if min_hits is None:
        min_hits = {"mid": 3, "high": 3}

    out = df.sort_values("timestamp").copy().reset_index(drop=True)
    out[out_col] = 0
    for z in ["mid", "high"]:
        m = out["ghi_zone"] == z
        if not m.any():
            continue
        w = int(windows.get(z, 5))
        h = int(min_hits.get(z, 3))
        rs = out.loc[m, pred_col].astype(int).rolling(w, min_periods=1).sum()
        out.loc[m, out_col] = (rs.values >= h).astype(int)
    return out


def sweep_fixed_threshold_by_zone(df, prob_col="attack_prob", zone="high", pa=True,
                                  thresholds=None, use_smoothing=False,
                                  smooth_window=5, smooth_min_hits=3, label=""):
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    print("\n" + "="*90)
    print(f"[DEBUG-2] Fixed threshold sweep | zone={zone} | PA={pa} | smoothing={use_smoothing} | {label}")
    print("="*90)
    sub = df[df["ghi_zone"] == zone].copy().reset_index(drop=True)
    if len(sub) == 0:
        print("  No data")
        return []

    rows = []
    for th in thresholds:
        tmp = sub.copy()
        tmp["sweep_pred"] = (tmp[prob_col] >= th).astype(int)
        pred_col = "sweep_pred"
        if use_smoothing:
            tmp = _apply_zone_smoothing(
                tmp,
                pred_col="sweep_pred",
                out_col="sweep_smooth_pred",
                windows={zone: smooth_window},
                min_hits={zone: smooth_min_hits},
            )
            pred_col = "sweep_smooth_pred"
        m = _evaluate_with_optional_pa(tmp, pred_col, f"{zone}@{th:.2f}", use_pa=pa)
        rows.append({"zone": zone, "threshold": th, **m})
        print(f"  th={th:>4.2f} | P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} FP={m['fp']:5d} FPR={m['fpr']*100:6.3f}% "
              f"DayFPR={m['day_fpr']:.4f}")
    return rows


def grid_search_zone_thresholds(df, prob_col="attack_prob", pa=True,
                                mid_grid=None, high_grid=None,
                                use_smoothing=False,
                                smooth_windows=None, smooth_min_hits=None,
                                top_k=30, label=""):
    if mid_grid is None:
        mid_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    if high_grid is None:
        high_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    if smooth_windows is None:
        smooth_windows = {"mid": 5, "high": 5}
    if smooth_min_hits is None:
        smooth_min_hits = {"mid": 3, "high": 3}

    print("\n" + "="*90)
    print(f"[DEBUG-3] Grid search fixed zone thresholds | PA={pa} | smoothing={use_smoothing} | {label}")
    print("="*90)

    rows = []
    for mt in mid_grid:
        for ht in high_grid:
            tmp = _apply_zone_fixed_threshold(df, prob_col, mt, ht)
            pred_col = "fixed_pred"
            if use_smoothing:
                tmp = _apply_zone_smoothing(
                    tmp,
                    pred_col="fixed_pred",
                    out_col="grid_smooth_pred",
                    windows=smooth_windows,
                    min_hits=smooth_min_hits,
                )
                pred_col = "grid_smooth_pred"
            m = _evaluate_with_optional_pa(tmp, pred_col, f"mid={mt:.2f}, high={ht:.2f}", use_pa=pa)
            rows.append({
                "mid_th": float(mt),
                "high_th": float(ht),
                "smooth": bool(use_smoothing),
                **m,
            })

    # F1 우선, 동률이면 FP와 DayFPR 낮은 순
    rows = sorted(rows, key=lambda r: (r["f1"], r["precision"], -r["fp"], -r["day_fpr"]), reverse=True)

    print(f"  Top {min(top_k, len(rows))} configs by PA-F1:")
    print("  rank | mid_th high_th | P      R      F1     FP    FPR%   DayR   DayFPR")
    for i, r in enumerate(rows[:top_k], 1):
        print(f"  {i:>4d} | {r['mid_th']:.2f}   {r['high_th']:.2f}   | "
              f"{r['precision']:.4f} {r['recall']:.4f} {r['f1']:.4f} "
              f"{r['fp']:>5d} {r['fpr']*100:>6.3f} {r['day_recall']:.4f} {r['day_fpr']:.4f}")
    return rows


def clean_day_fp_report(df, pred_col="final_pred", prob_col="attack_prob", zone="high", name=""):
    print("\n" + "="*90)
    print(f"[DEBUG-4] Clean-day false alarm pattern | zone={zone} | {name}")
    print("="*90)
    if pred_col not in df.columns:
        print(f"  ! Missing pred_col={pred_col}")
        return None
    sub = df[(df["ghi_zone"] == zone) & (df["is_attack"].astype(int) == 0)].copy()
    if len(sub) == 0:
        print("  No clean points")
        return None
    day = sub.groupby("date").agg(
        n_points=("timestamp", "count"),
        n_fp=(pred_col, "sum"),
        max_prob=(prob_col, "max"),
        mean_prob=(prob_col, "mean"),
    ).reset_index()
    bad = day[day["n_fp"] > 0].copy()
    print(f"  clean days={len(day)}, false-alarm days={len(bad)} ({len(bad)/len(day) if len(day) else 0:.4f})")
    if len(bad) > 0:
        print("  n_fp quantiles among false-alarm days:")
        print(bad["n_fp"].quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_string())
        print("\n  Top 20 false-alarm days:")
        print(bad.sort_values("n_fp", ascending=False).head(20).to_string(index=False))
    return day


def run_debug_tuning(f, tag):
    """10% SA에서 F1 0.9 가능성 확인용. predict_prob 이후, threshold 적용 전 f를 넣는다."""
    if not DEBUG_TUNING:
        return
    if DEBUG_ONLY_TAG_CONTAINS and DEBUG_ONLY_TAG_CONTAINS not in tag:
        return

    print("\n\n" + "#"*100)
    print(f"# DEBUG TUNING TARGET: {tag}")
    print("#"*100)

    debug_score_distribution(f, prob_col="attack_prob", pred_col=None, name=tag)

    # 모델 점수 자체가 분리되는지 확인. threshold 문제인지 model 문제인지 갈라짐.
    sweep_fixed_threshold_by_zone(f, prob_col="attack_prob", zone="high", pa=True, use_smoothing=False, label=tag)
    sweep_fixed_threshold_by_zone(f, prob_col="attack_prob", zone="high", pa=True, use_smoothing=True,
                                  smooth_window=5, smooth_min_hits=3, label=tag + " | high smooth 5/3")
    sweep_fixed_threshold_by_zone(f, prob_col="attack_prob", zone="mid", pa=True, use_smoothing=False, label=tag)

    # 전체 mid/high 고정 threshold grid. 최종 논문용이 아니라 가능성 확인용 oracle sweep.
    grid_search_zone_thresholds(
        f,
        prob_col="attack_prob",
        pa=True,
        mid_grid=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        high_grid=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        use_smoothing=False,
        top_k=DEBUG_TOP_K_GRID,
        label=tag + " | no smoothing",
    )
    grid_search_zone_thresholds(
        f,
        prob_col="attack_prob",
        pa=True,
        mid_grid=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        high_grid=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        use_smoothing=True,
        smooth_windows={"mid": 5, "high": 5},
        smooth_min_hits={"mid": 3, "high": 3},
        top_k=DEBUG_TOP_K_GRID,
        label=tag + " | smoothing 5/3",
    )

# =============================================================================
# 10. Main
# =============================================================================
def main():
    ensure_dir(OUT_DIR); ensure_dir(CACHE_DIR)
    drop_tag = f"drop={len(DROP_FEATURES)}" if DROP_FEATURES else "FULL"
    print("="*80)
    print(f"FADRE v3.11 — legacy decoupled + integrated threshold/F1 debug [{drop_tag}]")
    print("="*80)

    # 1. Forecaster
    stage_start("1 forecaster")
    fc=ForecasterWrapper().load()
    log("Residual centering/calibration disabled: using raw forecaster predictions only")
    stage_end("1 forecaster")

    # 2. Features
    stage_start("2 features")
    df_all=build_train_features(fc)
    ud=sorted(df_all["date"].unique()); si=int(len(ud)*TRAIN_RATIO)
    dtr=df_all[df_all["date"].isin(set(ud[:si]))].reset_index(drop=True)
    dva=df_all[df_all["date"].isin(set(ud[si:]))].reset_index(drop=True)
    log(f"train={len(dtr):,}(pos={int(dtr['is_attack'].sum())}) val={len(dva):,}(pos={int(dva['is_attack'].sum())})")
    stage_end("2 features")

    # 3. Train
    stage_start("3 train")
    zfc,drop=select_zone_features(dtr)
    dtr=dtr.drop(columns=drop,errors="ignore"); dva=dva.drop(columns=drop,errors="ignore")
    log(f"dropped {len(drop)} common features | high(raw+ghi)={len(zfc['high'])} mid(pr+ghi)={len(zfc['mid'])}")
    models,val_prob=train_zone_models(dtr,dva,zfc)
    stage_end("3 train")

    # 4. Validation
    stage_start("4 validation")
    val_dec=apply_final_fixed_threshold(val_prob)
    vm=evaluate(val_dec,"final_pred","validation")
    print(f"\n{'='*60}\n[4] Validation [{drop_tag}]\n{'='*60}")
    print(f"    Fixed threshold: mid={FIXED_PROB_THRESHOLD_BY_ZONE['mid']:.2f}, high={FIXED_PROB_THRESHOLD_BY_ZONE['high']:.2f}")
    pm("Validation",vm)
    for z in GHI_BINS:
        zd=val_dec[val_dec["ghi_zone"]==z]
        if len(zd)==0: continue
        zr=evaluate(zd,"final_pred",z)
        print(f"    [{z:>4s}] P={zr['precision']:.4f} R={zr['recall']:.4f} F1={zr['f1']:.4f} FP={zr['fp']}")
    stage_end("4 validation")

    # 5. SA Test
    stage_start("5 SA test")
    all_m={"validation":vm}
    print(f"\n{'='*60}\n[5] SA Test [{drop_tag}]\n{'='*60}")
    for path in EVAL_CSVS_SA:
        if not os.path.exists(path): continue
        tag=Path(path).stem
        r=load_residual_seq(path,fc,f"eval_{tag}"); f=extract_feat_cached(r,f"eval_{tag}")
        f=f[f["ghi_zone"]!="ignore"].reset_index(drop=True)
        f=f.drop(columns=drop,errors="ignore")
        f=predict_prob(f,models,zfc)

        if DEBUG_TUNING:
            run_debug_tuning(f, tag)

        f=apply_final_fixed_threshold(f)

        # fixed threshold가 실제로 어느 clean day에서 FP를 만드는지 확인한다.
        if DEBUG_TUNING and (not DEBUG_ONLY_TAG_CONTAINS or DEBUG_ONLY_TAG_CONTAINS in tag):
            clean_day_fp_report(f, pred_col="final_pred", prob_col="attack_prob", zone="high", name=tag + " | fixed final_pred")
            clean_day_fp_report(f, pred_col="final_pred", prob_col="attack_prob", zone="mid", name=tag + " | fixed final_pred")

        m=evaluate(f,"final_pred",tag); pm(tag,m)
        for z in GHI_BINS:
            zd=f[f["ghi_zone"]==z]
            if len(zd)==0: continue
            zr=evaluate(zd,"final_pred",z)
            print(f"    [{z:>4s}] P={zr['precision']:.4f} R={zr['recall']:.4f} F1={zr['f1']:.4f} FP={zr['fp']}")
        all_m[tag]=m
    stage_end("5 SA test")

    # 6. Summary + Importance
    print(f"\n{'='*60}\n[6] 요약 [{drop_tag}]\n{'='*60}")
    print(f"  {'Dataset':<45s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'FP':>6s} {'DayR':>6s} {'DayFPR':>6s}")
    for k,m in all_m.items():
        print(f"  {k:<45s} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f} {m['fp']:>6d} {m['day_recall']:>6.4f} {m['day_fpr']:>6.4f}")

    print(f"\n  Feature Importance (전체 → 바닥권 찾아서 DROP_FEATURES에 추가)")
    for z,clf in models.items():
        imp=pd.Series(clf.feature_importances_,index=zfc[z]).sort_values(ascending=False)
        print(f"\n  [{z.upper()}] (총 {len(imp)}개)")
        for r,(fn,fv) in enumerate(imp.items(),1):
            bar="█"*int(fv/max(imp.max(),1)*20)
            marker=" ← 제거 후보" if fv<20 else ""
            print(f"    {r:3d}. {fn:35s} {fv:5.0f} {bar}{marker}")

    # Save
    with open(os.path.join(OUT_DIR,"metrics_iter.json"),"w") as f:
        json.dump(all_m,f,indent=2,ensure_ascii=False)
    log("DONE")


if __name__=="__main__":
    main()
