# 03_xgboost_model_fixed_calibrated.py
"""
Thermal intensity modeling pipeline (OLS / Random Forest / XGBoost)
--- Extended with:
 - Target transformation (none / log1p / Yeo-Johnson)
 - Post-hoc calibration on validation (I_obs = a + b * I_pred)
 - Reporting: metrics before/after calibration, slope/intercept, bias reduction
 - Error reporting by deciles of I and by climate/HDD bins
Author: Fafa (modified)
"""
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


# ----------------------------
# Small helper: compatible OneHotEncoder
# ----------------------------
def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ----------------------------
# Target transformer (wraps none/log/yeo)
# ----------------------------
class TargetTransformer:
    """
    Provides fit/transform/inverse_transform interface for target y.
    kinds: "none", "log1p", "yeo"
    """
    def __init__(self, kind: str = "none"):
        kind = kind.lower()
        if kind not in ("none", "log1p", "yeo"):
            raise ValueError("kind must be one of 'none','log1p','yeo'")
        self.kind = kind
        self._pt = None  # for yeo
        self.fitted = False

    def fit(self, y: np.ndarray):
        y = np.asarray(y).reshape(-1, 1).astype(float)
        if self.kind == "yeo":
            self._pt = PowerTransformer(method="yeo-johnson", standardize=False)
            self._pt.fit(y)
        # log1p and none need no fit
        self.fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1, 1).astype(float)
        if self.kind == "none":
            return y.ravel()
        elif self.kind == "log1p":
            return np.log1p(y).ravel()
        elif self.kind == "yeo":
            return self._pt.transform(y).ravel()

    def inverse_transform(self, y_trans: np.ndarray) -> np.ndarray:
        y_trans = np.asarray(y_trans).reshape(-1, 1).astype(float)
        if self.kind == "none":
            return y_trans.ravel()
        elif self.kind == "log1p":
            return np.expm1(y_trans).ravel()
        elif self.kind == "yeo":
            return self._pt.inverse_transform(y_trans).ravel()


# ----------------------------
# Model wrapper to keep preprocessing + estimator + target-inverse together
# ----------------------------
@dataclass
class PreprocessedRegressor:
    preprocessor: ColumnTransformer  # must be already fit
    model: Any  # estimator (fit on transformed target)
    feature_names_: List[str]
    y_inverse_fn: Callable[[np.ndarray], np.ndarray]  # function to inverse-transform model outputs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.preprocessor.transform(X)
        yp = self.model.predict(Xt)
        # ensure shape is (n,)
        yp = np.asarray(yp).ravel()
        return self.y_inverse_fn(yp)


# ----------------------------
# Data loading & features
# ----------------------------
def load_processed_data() -> pd.DataFrame:
    filepath = OUTPUT_DIR / "03_gas_heated_clean.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. Run 01_data_prep.py first."
        )
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows from {filepath}")
    return df


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = [
        "HDD65",
        "A_heated",
        "building_age",
        "heating_equip_age",
        "log_sqft",
    ]
    categorical_features = [
        "TYPEHUQ",        # housing type
        "YEARMADERANGE",  # year built category code
        "DRAFTY",
        "ADQINSUL",
        "TYPEGLASS",
        "EQUIPM",
        "REGIONC",
        "DIVISION",
        "envelope_class",
        "climate_zone",
    ]

    num_avail = [c for c in numeric_features if c in df.columns]
    cat_avail = [c for c in categorical_features if c in df.columns]

    return num_avail, cat_avail


def prepare_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "Thermal_Intensity_I" not in df.columns:
        raise KeyError("Target 'Thermal_Intensity_I' not found. Run 01_data_prep.py first.")

    num_cols, cat_cols = get_feature_lists(df)

    X = df[num_cols + cat_cols].copy()
    y = df["Thermal_Intensity_I"].copy()

    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    for c in cat_cols:
        X[c] = X[c].astype("object")

    logger.info(f"Prepared X,y with {X.shape[0]:,} samples and {X.shape[1]} raw features.")
    return X, y


# ----------------------------
# Splitting (unchanged)
# ----------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify_col: str = "REGIONC",
) -> Tuple:
    df_sub = df_full.loc[X.index].copy()
    weights = df_sub["NWEIGHT"].values if "NWEIGHT" in df_sub.columns else None

    strat = None
    if stratify_col in df_sub.columns:
        strat = df_sub[stratify_col].fillna("missing").astype(str)
        logger.info(f"Using '{stratify_col}' for stratified splitting.")
    else:
        logger.warning(f"'{stratify_col}' not found -> no stratification.")

    if weights is not None:
        if strat is not None:
            X_trainval, X_test, y_trainval, y_test, strat_trainval, strat_test, w_trainval, w_test = train_test_split(
                X, y, strat, weights,
                test_size=test_size,
                random_state=42,
                stratify=strat,
            )
        else:
            X_trainval, X_test, y_trainval, y_test, w_trainval, w_test = train_test_split(
                X, y, weights,
                test_size=test_size,
                random_state=42,
            )
            strat_trainval = None
    else:
        if strat is not None:
            X_trainval, X_test, y_trainval, y_test, strat_trainval, strat_test = train_test_split(
                X, y, strat,
                test_size=test_size,
                random_state=42,
                stratify=strat,
            )
        else:
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
            )
            strat_trainval = None
        w_trainval = w_test = None

    adjusted_val_size = val_size / (1.0 - test_size)

    if weights is not None:
        if strat_trainval is not None:
            X_train, X_val, y_train, y_val, strat_train, strat_val, w_train, w_val = train_test_split(
                X_trainval, y_trainval, strat_trainval, w_trainval,
                test_size=adjusted_val_size,
                random_state=42,
                stratify=strat_trainval,
            )
        else:
            X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
                X_trainval, y_trainval, w_trainval,
                test_size=adjusted_val_size,
                random_state=42,
            )
    else:
        if strat_trainval is not None:
            X_train, X_val, y_train, y_val, strat_train, strat_val = train_test_split(
                X_trainval, y_trainval, strat_trainval,
                test_size=adjusted_val_size,
                random_state=42,
                stratify=strat_trainval,
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=adjusted_val_size,
                random_state=42,
            )
        w_train = w_val = None

    df_train = df_sub.loc[X_train.index]
    df_val = df_sub.loc[X_val.index]
    df_test = df_sub.loc[X_test.index]

    logger.info(f"Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        w_train, w_val, w_test,
        df_train, df_val, df_test,
    )


# ----------------------------
# Preprocessor
# ----------------------------
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", make_onehot_encoder()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out().tolist()
        return [str(n) for n in names]
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform(pd.DataFrame()).shape[1])]


# ----------------------------
# Metrics (unchanged)
# ----------------------------
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out = {
        "n_samples": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _safe_mape(y_true, y_pred),
        "bias_mean": float(np.mean(y_true - y_pred)),            # mean bias (obs - pred)
        "bias_abs_mean": float(np.mean(np.abs(y_true - y_pred))), # mean absolute bias
    }

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        if np.any(w < 0):
            raise ValueError("Sample weights contain negative values.")
        if w.sum() <= 0:
            out.update({"weighted_rmse": np.nan, "weighted_mae": np.nan,
                        "weighted_r2": np.nan, "weighted_mape": np.nan})
            return out

        w = w / w.sum()

        sq_err = (y_true - y_pred) ** 2
        abs_err = np.abs(y_true - y_pred)

        out["weighted_rmse"] = float(np.sqrt(np.average(sq_err, weights=w)))
        out["weighted_mae"] = float(np.average(abs_err, weights=w))

        y_bar = np.average(y_true, weights=w)
        sse = np.average((y_true - y_pred) ** 2, weights=w)
        tss = np.average((y_true - y_bar) ** 2, weights=w)
        out["weighted_r2"] = float(1.0 - sse / tss) if tss > 0 else np.nan

        eps = 1e-8
        denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
        out["weighted_mape"] = float(np.average(np.abs((y_true - y_pred) / denom), weights=w) * 100.0)

    return out


def evaluate_model(
    model_obj: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray] = None,
    set_name: str = "Test",
) -> Dict[str, float]:
    y_pred = model_obj.predict(X)
    metrics = evaluate_predictions(y.values, y_pred, sample_weight=sample_weight)

    logger.info(f"{set_name} (unweighted): RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")
    if sample_weight is not None and "weighted_rmse" in metrics:
        logger.info(f"{set_name} (weighted):   RMSE_w={metrics['weighted_rmse']:.4f}, MAE_w={metrics['weighted_mae']:.4f}, R²_w={metrics['weighted_r2']:.4f}, MAPE_w={metrics['weighted_mape']:.2f}%")

    metrics["set"] = set_name
    return metrics


# ----------------------------
# Training helpers (fit preprocessor + model on transformed targets, return PreprocessedRegressor)
# ----------------------------
def train_ols(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_transformer: TargetTransformer,
    sample_weight: Optional[np.ndarray] = None,
) -> PreprocessedRegressor:
    # fit preprocessor on train
    preprocessor.fit(X_train)
    Xtr = preprocessor.transform(X_train)
    # transform target
    target_transformer.fit(y_train.values)
    ytr = target_transformer.transform(y_train.values)

    model = LinearRegression()
    if sample_weight is not None:
        model.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        model.fit(Xtr, ytr)

    feature_names = get_feature_names(preprocessor)
    return PreprocessedRegressor(preprocessor=preprocessor, model=model, feature_names_=feature_names, y_inverse_fn=target_transformer.inverse_transform)


def train_random_forest(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_transformer: TargetTransformer,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> PreprocessedRegressor:
    default = dict(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    if params:
        default.update(params)

    preprocessor.fit(X_train)
    Xtr = preprocessor.transform(X_train)
    target_transformer.fit(y_train.values)
    ytr = target_transformer.transform(y_train.values)

    model = RandomForestRegressor(**default)
    if sample_weight is not None:
        model.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        model.fit(Xtr, ytr)

    feature_names = get_feature_names(preprocessor)
    return PreprocessedRegressor(preprocessor=preprocessor, model=model, feature_names_=feature_names, y_inverse_fn=target_transformer.inverse_transform)


def train_xgboost(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_transformer: TargetTransformer,
    sample_weight: Optional[np.ndarray] = None,
    sample_weight_val: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> PreprocessedRegressor:
    base_params = dict(
        n_estimators=2000,          # with early stopping this is safe
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    if params:
        base_params.update(params)

    # fit preprocessor on TRAIN only
    preprocessor.fit(X_train)
    Xtr = preprocessor.transform(X_train)
    Xva = preprocessor.transform(X_val)

    # target transform
    target_transformer.fit(y_train.values)
    ytr = target_transformer.transform(y_train.values)
    yva = target_transformer.transform(y_val.values)

    feature_names = get_feature_names(preprocessor)

    model = xgb.XGBRegressor(**base_params)

    fit_kwargs = dict()
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    try:
        if sample_weight_val is not None:
            fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False,
            **fit_kwargs,
        )
        logger.info("XGBoost trained with early stopping.")
    except TypeError:
        model = xgb.XGBRegressor(**{k: v for k, v in base_params.items() if k != "n_estimators"} | {"n_estimators": 400})
        if sample_weight is not None:
            model.fit(Xtr, ytr, sample_weight=sample_weight)
        else:
            model.fit(Xtr, ytr)
        logger.warning("XGBoost early stopping not supported; trained with fixed n_estimators=400.")
    except Exception as e:
        model = xgb.XGBRegressor(**{k: v for k, v in base_params.items() if k != "n_estimators"} | {"n_estimators": 400})
        if sample_weight is not None:
            model.fit(Xtr, ytr, sample_weight=sample_weight)
        else:
            model.fit(Xtr, ytr)
        logger.warning(f"XGBoost early stopping failed ({e}); trained with fixed n_estimators=400.")

    return PreprocessedRegressor(preprocessor=preprocessor, model=model, feature_names_=feature_names, y_inverse_fn=target_transformer.inverse_transform)


# ----------------------------
# Subgroup evaluation (unchanged except using model.predict wrapper)
# ----------------------------
def evaluate_by_subgroups(
    model_obj: Any,
    X: pd.DataFrame,
    y: pd.Series,
    df_sub: pd.DataFrame,
    groupby_cols: List[str],
    sample_weight: Optional[np.ndarray] = None,
    min_n: int = 30,
) -> pd.DataFrame:
    y_pred = pd.Series(model_obj.predict(X), index=X.index)

    results: List[Dict] = []

    for col in groupby_cols:
        if col not in df_sub.columns:
            logger.warning(f"Group column '{col}' not found; skipping.")
            continue

        groups = df_sub[col]
        for g in groups.dropna().unique():
            mask = (groups == g)
            n = int(mask.sum())
            if n < min_n:
                continue

            y_g = y.loc[mask]
            yhat_g = y_pred.loc[mask].values

            w_g = None
            if sample_weight is not None:
                w_g = np.asarray(sample_weight)[mask.values]

            m = evaluate_predictions(y_g.values, yhat_g, sample_weight=w_g)
            results.append({
                "Group Variable": col,
                "Group": g,
                "N": n,
                "RMSE": m["rmse"],
                "MAE": m["mae"],
                "R²": m["r2"],
                "RMSE_w": m.get("weighted_rmse", np.nan),
                "R²_w": m.get("weighted_r2", np.nan),
                "Bias_mean": m.get("bias_mean", np.nan),
            })

    return pd.DataFrame(results)


# ----------------------------
# Calibration: fit on validation, apply on test
# ----------------------------
def fit_posthoc_calibration(y_val: np.ndarray, yval_pred: np.ndarray) -> Tuple[float, float]:
    """
    Fit linear regression: y_val = a + b * yval_pred
    Return (a, b)
    """
    lr = LinearRegression()
    X = np.asarray(yval_pred).reshape(-1, 1)
    y = np.asarray(y_val).reshape(-1, 1)
    lr.fit(X, y)
    a = float(lr.intercept_.ravel()[0])
    b = float(lr.coef_.ravel()[0])
    logger.info(f"Calibration fitted on validation: intercept(a)={a:.4f}, slope(b)={b:.4f}")
    return a, b


def apply_calibration(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.asarray(y_pred)


# ----------------------------
# Error by deciles / climate / HDD bins
# ----------------------------
def error_by_deciles_and_climate(
    y_true: pd.Series,
    y_pred_uncal: np.ndarray,
    y_pred_calib: Optional[np.ndarray],
    df_test: pd.DataFrame,
    sample_weight: Optional[np.ndarray] = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Produce table with metrics per decile of observed I, and per climate_zone, and HDD65 bins.
    Returns concatenated DataFrame (decile-level + climate-level + HDD-bin-level).
    """
    out_rows = []

    # Deciles of observed I
    decile_labels = [f"D{d}" for d in range(1, n_deciles + 1)]
    try:
        deciles = pd.qcut(y_true.rank(method="first"), q=n_deciles, labels=decile_labels)
    except Exception:
        # fallback: equal-frequency on raw
        deciles = pd.qcut(y_true, q=n_deciles, labels=decile_labels, duplicates="drop")

    for d in deciles.unique().dropna():
        mask = (deciles == d)
        n = int(mask.sum())
        if n < 1:
            continue
        y_g = y_true[mask]
        pred_unc = np.asarray(y_pred_uncal)[mask.values]
        pred_cal = np.asarray(y_pred_calib)[mask.values] if y_pred_calib is not None else None
        w_g = None
        if sample_weight is not None:
            w_g = np.asarray(sample_weight)[mask.values]
        m_unc = evaluate_predictions(y_g.values, pred_unc, sample_weight=w_g)
        row = {"GroupType": "Decile_I", "Group": d, "N": n, "RMSE_uncal": m_unc["rmse"], "MAE_uncal": m_unc["mae"], "Bias_mean_uncal": m_unc["bias_mean"]}
        if pred_cal is not None:
            m_cal = evaluate_predictions(y_g.values, pred_cal, sample_weight=w_g)
            row.update({"RMSE_cal": m_cal["rmse"], "MAE_cal": m_cal["mae"], "Bias_mean_cal": m_cal["bias_mean"]})
        out_rows.append(row)

    # climate_zone
    if "climate_zone" in df_test.columns:
        for g in df_test["climate_zone"].dropna().unique():
            mask = (df_test["climate_zone"] == g)
            n = int(mask.sum())
            if n < 1:
                continue
            y_g = y_true[mask]
            pred_unc = np.asarray(y_pred_uncal)[mask.values]
            pred_cal = np.asarray(y_pred_calib)[mask.values] if y_pred_calib is not None else None
            w_g = None
            if sample_weight is not None:
                w_g = np.asarray(sample_weight)[mask.values]
            m_unc = evaluate_predictions(y_g.values, pred_unc, sample_weight=w_g)
            row = {"GroupType": "Climate", "Group": g, "N": n, "RMSE_uncal": m_unc["rmse"], "MAE_uncal": m_unc["mae"], "Bias_mean_uncal": m_unc["bias_mean"]}
            if pred_cal is not None:
                m_cal = evaluate_predictions(y_g.values, pred_cal, sample_weight=w_g)
                row.update({"RMSE_cal": m_cal["rmse"], "MAE_cal": m_cal["mae"], "Bias_mean_cal": m_cal["bias_mean"]})
            out_rows.append(row)

    # HDD65 bins (tertiles)
    if "HDD65" in df_test.columns:
        try:
            hdd_bins = pd.qcut(df_test["HDD65"].rank(method="first"), q=3, labels=["H_low", "H_med", "H_high"])
        except Exception:
            hdd_bins = pd.qcut(df_test["HDD65"], q=3, labels=["H_low", "H_med", "H_high"], duplicates="drop")
        for g in hdd_bins.unique().dropna():
            mask = (hdd_bins == g)
            n = int(mask.sum())
            if n < 1:
                continue
            y_g = y_true[mask]
            pred_unc = np.asarray(y_pred_uncal)[mask.values]
            pred_cal = np.asarray(y_pred_calib)[mask.values] if y_pred_calib is not None else None
            w_g = None
            if sample_weight is not None:
                w_g = np.asarray(sample_weight)[mask.values]
            m_unc = evaluate_predictions(y_g.values, pred_unc, sample_weight=w_g)
            row = {"GroupType": "HDD_bin", "Group": g, "N": n, "RMSE_uncal": m_unc["rmse"], "MAE_uncal": m_unc["mae"], "Bias_mean_uncal": m_unc["bias_mean"]}
            if pred_cal is not None:
                m_cal = evaluate_predictions(y_g.values, pred_cal, sample_weight=w_g)
                row.update({"RMSE_cal": m_cal["rmse"], "MAE_cal": m_cal["mae"], "Bias_mean_cal": m_cal["bias_mean"]})
            out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    return df_out


# ----------------------------
# Figure and table generation (slightly extended to include calibration info)
# ----------------------------
def generate_figure5_predictions(
    y_true: pd.Series,
    y_pred_rf: np.ndarray,
    y_pred_xgb: np.ndarray,
    groups: Optional[pd.Series] = None,
    group_name: str = "Division",
    sample_weight: Optional[np.ndarray] = None,
):
    logger.info("Generating Figure 5: Predicted vs observed (RF & XGBoost)")

    y_true_arr = np.asarray(y_true).astype(float)
    y_pred_rf = np.asarray(y_pred_rf).astype(float)
    y_pred_xgb = np.asarray(y_pred_xgb).astype(float)

    w = None
    if sample_weight is not None:
        w = np.asarray(sample_weight).astype(float)
        if hasattr(sample_weight, "index") and hasattr(y_true, "index"):
            try:
                w = np.asarray(sample_weight.reindex(y_true.index)).astype(float)
            except Exception:
                pass
        w = np.where(np.isfinite(w), w, 0.0)
        w_sum = w.sum()
        w = (w / w_sum) if w_sum > 0 else None

    all_vals = np.concatenate([y_true_arr, y_pred_rf, y_pred_xgb])
    min_val = float(np.nanmin(all_vals))
    max_val = float(np.nanmax(all_vals))
    padding = 0.5
    lims = [max(min_val - padding, 0.0), max_val + padding]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    def _weighted_r2(y_t: np.ndarray, y_p: np.ndarray, w_: np.ndarray) -> float:
        y_bar = np.average(y_t, weights=w_)
        sse = np.average((y_t - y_p) ** 2, weights=w_)
        tss = np.average((y_t - y_bar) ** 2, weights=w_)
        return 1.0 - sse / tss if tss > 0 else np.nan

    def _weighted_rmse(y_t: np.ndarray, y_p: np.ndarray, w_: np.ndarray) -> float:
        return float(np.sqrt(np.average((y_t - y_p) ** 2, weights=w_)))

    def _calibration(y_t: np.ndarray, y_p: np.ndarray, w_: Optional[np.ndarray]) -> Tuple[float, float]:
        mask = np.isfinite(y_t) & np.isfinite(y_p)
        if w_ is not None:
            mask = mask & np.isfinite(w_) & (w_ > 0)
        if mask.sum() < 3:
            return (np.nan, np.nan)
        if w_ is None:
            a, b = np.polyfit(y_t[mask], y_p[mask], 1)
        else:
            a, b = np.polyfit(y_t[mask], y_p[mask], 1, w=w_[mask])
        return float(a), float(b)

    def _plot(ax, y_pred: np.ndarray, title: str):
        if groups is not None:
            groups_local = groups.reindex(y_true.index)
            for g in groups_local.dropna().unique():
                mask_g = (groups_local == g).values
                ax.scatter(
                    y_true_arr[mask_g],
                    y_pred[mask_g],
                    alpha=0.45,
                    s=18,
                    label=str(g),
                )
        else:
            ax.scatter(y_true_arr, y_pred, alpha=0.45, s=18)

        ax.plot(lims, lims, "k--", alpha=0.75, zorder=0, linewidth=2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        r2 = r2_score(y_true_arr, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred))
        a, b = _calibration(y_true_arr, y_pred, None)

        lines = [
            f"R² = {r2:.3f}",
            f"RMSE = {rmse:.2f}",
            f"slope = {a:.2f}, intercept = {b:.2f}",
        ]

        if w is not None:
            r2_w = _weighted_r2(y_true_arr, y_pred, w)
            rmse_w = _weighted_rmse(y_true_arr, y_pred, w)
            a_w, b_w = _calibration(y_true_arr, y_pred, w)
            lines += [
                f"R²_w = {r2_w:.3f}",
                f"RMSE_w = {rmse_w:.2f}",
                f"slope_w = {a_w:.2f}, intercept_w = {b_w:.2f}",
            ]

        ax.annotate(
            "\n".join(lines),
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.set_xlabel("Observed thermal intensity, I (BTU/ft²·HDD)", fontsize=12)
        ax.set_ylabel("Predicted thermal intensity, I (BTU/ft²·HDD)", fontsize=12)
        ax.set_title(title, fontsize=14)

    _plot(axes[0], y_pred_rf, "(a) Random Forest model")
    _plot(axes[1], y_pred_xgb, "(b) XGBoost model")

    if groups is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, title=group_name, fontsize=9, title_fontsize=10, frameon=True, loc="lower right")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "figure5_predicted_vs_observed.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure5_predicted_vs_observed.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_table3_model_performance(
    rows: List[Dict],
    subgroup_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    logger.info("Generating Table 3: model performance (train/val/test)")
    table = pd.DataFrame(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "table3_model_performance.csv", index=False)

    try:
        table.to_latex(
            TABLES_DIR / "table3_model_performance.tex",
            index=False,
            float_format="%.3f",
            caption=(
                "Performance of OLS, Random Forest and XGBoost models for predicting "
                "residential space-heating thermal intensity on train, validation and test sets."
            ),
            label="tab:model_performance",
        )
    except Exception as e:
        logger.warning(f"Could not write LaTeX Table 3: {e}")

    if subgroup_metrics is not None and not subgroup_metrics.empty:
        subgroup_metrics.to_csv(TABLES_DIR / "table3_subgroup_performance_rf.csv", index=False)

    return table


# ----------------------------
# Cross-validation (unchanged)
# ----------------------------
def cross_validate_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    n_splits: int = 5,
    stratify_col: str = "REGIONC",
) -> Dict[str, float]:
    df_sub = df_full.loc[X.index]
    weights = df_sub["NWEIGHT"].values if "NWEIGHT" in df_sub.columns else None

    if stratify_col in df_sub.columns:
        strat = df_sub[stratify_col].fillna("missing").astype(str).values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = skf.split(X, strat)
        logger.info(f"CV: StratifiedKFold({n_splits}) on '{stratify_col}' (shuffle=True).")
    else:
        y_bins = pd.qcut(y.rank(method="first"), q=min(10, max(2, n_splits)), duplicates="drop").astype(str).values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = skf.split(X, y_bins)
        logger.warning(f"'{stratify_col}' not found -> CV stratified by binned target ranks.")

    fold_r2 = []
    fold_rmse = []
    fold_r2_w = []
    fold_rmse_w = []

    for k, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        wtr = weights[tr_idx] if weights is not None else None
        wte = weights[te_idx] if weights is not None else None

        pre = build_preprocessor(num_cols, cat_cols)
        pre.fit(Xtr)
        Xtr_t = pre.transform(Xtr)
        Xte_t = pre.transform(Xte)

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        )

        if wtr is not None:
            model.fit(Xtr_t, ytr, sample_weight=wtr)
        else:
            model.fit(Xtr_t, ytr)

        yhat = model.predict(Xte_t)
        m = evaluate_predictions(yte.values, yhat, sample_weight=wte)

        fold_r2.append(m["r2"])
        fold_rmse.append(m["rmse"])
        fold_r2_w.append(m.get("weighted_r2", np.nan))
        fold_rmse_w.append(m.get("weighted_rmse", np.nan))

        logger.info(
            f"CV fold {k}/{n_splits}: R²={m['r2']:.4f}, RMSE={m['rmse']:.4f}"
            + (f", R²_w={m.get('weighted_r2', np.nan):.4f}, RMSE_w={m.get('weighted_rmse', np.nan):.4f}" if weights is not None else "")
        )

    out = {
        "r2_mean": float(np.nanmean(fold_r2)),
        "r2_std": float(np.nanstd(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0,
        "rmse_mean": float(np.nanmean(fold_rmse)),
        "rmse_std": float(np.nanstd(fold_rmse, ddof=1)) if len(fold_rmse) > 1 else 0.0,
    }
    if weights is not None:
        out.update({
            "weighted_r2_mean": float(np.nanmean(fold_r2_w)),
            "weighted_r2_std": float(np.nanstd(fold_r2_w, ddof=1)) if len(fold_r2_w) > 1 else 0.0,
            "weighted_rmse_mean": float(np.nanmean(fold_rmse_w)),
            "weighted_rmse_std": float(np.nanstd(fold_rmse_w, ddof=1)) if len(fold_rmse_w) > 1 else 0.0,
        })

    logger.info(f"CV summary: R²={out['r2_mean']:.4f} ± {out['r2_std']:.4f}, RMSE={out['rmse_mean']:.4f} ± {out['rmse_std']:.4f}")
    if weights is not None:
        logger.info(f"CV summary (weighted): R²_w={out['weighted_r2_mean']:.4f} ± {out['weighted_r2_std']:.4f}, RMSE_w={out['weighted_rmse_mean']:.4f} ± {out['weighted_rmse_std']:.4f}")

    return out


# ----------------------------
# Main pipeline (with new options)
# ----------------------------
def run_modeling_pipeline(target_transform: str = "none"):
    """
    target_transform: "none" | "log1p" | "yeo"
    """
    logger.info("=" * 60)
    logger.info("Thermal intensity modeling pipeline (OLS / RF / XGBoost) - extended")
    logger.info("=" * 60)
    logger.info(f"Versions: xgboost={xgb.__version__}")
    logger.info(f"Target transform: {target_transform}")

    df = load_processed_data()
    X, y = prepare_X_y(df)
    num_cols, cat_cols = get_feature_lists(df)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        w_train, w_val, w_test,
        df_train, df_val, df_test,
    ) = split_data(X, y, df, stratify_col="REGIONC")

    # Cross-validation (weighted) -- unchanged use case
    cv_results = cross_validate_xgb(X, y, df, num_cols=num_cols, cat_cols=cat_cols, n_splits=5, stratify_col="REGIONC")

    # Build preprocessors
    pre_ols = build_preprocessor(num_cols, cat_cols)
    pre_rf = build_preprocessor(num_cols, cat_cols)
    pre_xgb = build_preprocessor(num_cols, cat_cols)

    # Target transformer
    tgt = TargetTransformer(kind=target_transform)

    # 1) OLS baseline
    model_ols = train_ols(pre_ols, X_train, y_train, tgt, sample_weight=w_train)

    # 2) Random Forest (main model)
    model_rf = train_random_forest(pre_rf, X_train, y_train, tgt, sample_weight=w_train)

    # 3) XGBoost (benchmark)
    model_xgb = train_xgboost(pre_xgb, X_train, y_train, X_val, y_val, tgt, sample_weight=w_train, sample_weight_val=w_val)

    # Evaluate all sets (models return predictions in original units via y_inverse_fn)
    ols_train = evaluate_model(model_ols, X_train, y_train, w_train, "Train (OLS)")
    ols_val = evaluate_model(model_ols, X_val, y_val, w_val, "Val (OLS)")
    ols_test = evaluate_model(model_ols, X_test, y_test, w_test, "Test (OLS)")

    rf_train = evaluate_model(model_rf, X_train, y_train, w_train, "Train (RF)")
    rf_val = evaluate_model(model_rf, X_val, y_val, w_val, "Val (RF)")
    rf_test = evaluate_model(model_rf, X_test, y_test, w_test, "Test (RF)")

    xgb_train = evaluate_model(model_xgb, X_train, y_train, w_train, "Train (XGB)")
    xgb_val = evaluate_model(model_xgb, X_val, y_val, w_val, "Val (XGB)")
    xgb_test = evaluate_model(model_xgb, X_test, y_test, w_test, "Test (XGB)")

    # Subgroup performance (RF main model)
    subgroup_metrics = evaluate_by_subgroups(
        model_rf,
        X_test,
        y_test,
        df_test,
        groupby_cols=["division_name", "envelope_class", "climate_zone"],
        sample_weight=w_test,
        min_n=30,
    )
    if not subgroup_metrics.empty:
        subgroup_metrics.to_csv(TABLES_DIR / "table3_subgroup_performance_rf.csv", index=False)

    # Figure 5
    y_pred_test_rf = model_rf.predict(X_test)
    y_pred_test_xgb = model_xgb.predict(X_test)
    groups = df_test["division_name"] if "division_name" in df_test.columns else None
    generate_figure5_predictions(y_test, y_pred_test_rf, y_pred_test_xgb, groups, sample_weight=w_test)

    # ----------------------------
    # Post-hoc calibration (fit on validation predictions)
    # ----------------------------
    logger.info("Fitting post-hoc calibration on validation set (I_obs = a + b * I_pred).")
    # Use the primary model for calibration (Random Forest). Could be done for others similarly.
    yval_pred_rf = model_rf.predict(X_val)
    a_cal, b_cal = fit_posthoc_calibration(y_val.values, yval_pred_rf)

    # Apply calibration to test preds
    ypred_test_rf_uncal = y_pred_test_rf
    ypred_test_rf_cal = apply_calibration(ypred_test_rf_uncal, a_cal, b_cal)

    # Compute slope/intercept on test before/after calibration (regress y_test ~ y_pred)
    def compute_slope_intercept(y_true_arr, y_pred_arr):
        lr = LinearRegression()
        lr.fit(np.asarray(y_pred_arr).reshape(-1, 1), np.asarray(y_true_arr).reshape(-1, 1))
        a = float(lr.intercept_.ravel()[0])
        b = float(lr.coef_.ravel()[0])
        return a, b

    a_before, b_before = compute_slope_intercept(y_test.values, ypred_test_rf_uncal)
    a_after, b_after = compute_slope_intercept(y_test.values, ypred_test_rf_cal)

    # Evaluate calibrated predictions
    metrics_uncal = evaluate_predictions(y_test.values, ypred_test_rf_uncal, sample_weight=w_test)
    metrics_cal = evaluate_predictions(y_test.values, ypred_test_rf_cal, sample_weight=w_test)

    logger.info("Calibration results on Test set (Random Forest):")
    logger.info(f"  Before calib: slope={b_before:.4f}, intercept={a_before:.4f}, bias_mean={metrics_uncal['bias_mean']:.4f}")
    logger.info(f"  After  calib: slope={b_after:.4f}, intercept={a_after:.4f}, bias_mean={metrics_cal['bias_mean']:.4f}")

    # Save calibration summary
    calib_summary = {
        "model": "RandomForest",
        "calib_intercept_val_a": a_cal,
        "calib_slope_val_b": b_cal,
        "test_slope_before": b_before,
        "test_intercept_before": a_before,
        "test_slope_after": b_after,
        "test_intercept_after": a_after,
        "bias_mean_before": metrics_uncal["bias_mean"],
        "bias_mean_after": metrics_cal["bias_mean"],
        "RMSE_before": metrics_uncal["rmse"],
        "RMSE_after": metrics_cal["rmse"],
        "MAE_before": metrics_uncal["mae"],
        "MAE_after": metrics_cal["mae"],
        "MAPE_before": metrics_uncal["mape"],
        "MAPE_after": metrics_cal["mape"],
    }
    pd.DataFrame([calib_summary]).to_csv(TABLES_DIR / "calibration_summary_rf.csv", index=False)

    # Detailed error by deciles/climate/HDD
    df_error_breakdown = error_by_deciles_and_climate(y_test, ypred_test_rf_uncal, ypred_test_rf_cal, df_test, sample_weight=w_test)
    df_error_breakdown.to_csv(TABLES_DIR / "error_by_decile_climate_hdd_rf.csv", index=False)

    # ----------------------------
    # Table 3 rows (same as before)
    rows: List[Dict] = []

    def add_row(model_name: str, set_name: str, m: Dict[str, float]):
        row = {
            "Model": model_name,
            "Set": set_name,
            "n_samples": m.get("n_samples", np.nan),
            "RMSE": m.get("rmse", np.nan),
            "MAE": m.get("mae", np.nan),
            "R²": m.get("r2", np.nan),
            "MAPE": m.get("mape", np.nan),
        }
        if "weighted_rmse" in m:
            row["RMSE_w"] = m.get("weighted_rmse", np.nan)
            row["MAE_w"] = m.get("weighted_mae", np.nan)
            row["R²_w"] = m.get("weighted_r2", np.nan)
            row["MAPE_w"] = m.get("weighted_mape", np.nan)
        rows.append(row)

    for set_name, m in [("Train", xgb_train), ("Validation", xgb_val), ("Test", xgb_test)]:
        add_row("XGBoost", set_name, m)
    for set_name, m in [("Train", ols_train), ("Validation", ols_val), ("Test", ols_test)]:
        add_row("OLS", set_name, m)
    for set_name, m in [("Train", rf_train), ("Validation", rf_val), ("Test", rf_test)]:
        add_row("Random Forest", set_name, m)

    model_perf_df = generate_table3_model_performance(rows, subgroup_metrics=subgroup_metrics)

    # Test-only comparison table (same columns as before)
    comparison_rows = []
    for name, m in [("OLS", ols_test), ("Random Forest", rf_test), ("XGBoost", xgb_test)]:
        row = {
            "Model": name,
            "RMSE": m.get("rmse", np.nan),
            "MAE": m.get("mae", np.nan),
            "R²": m.get("r2", np.nan),
            "MAPE": m.get("mape", np.nan),
        }
        if "weighted_rmse" in m:
            row["RMSE_w"] = m.get("weighted_rmse", np.nan)
            row["MAE_w"] = m.get("weighted_mae", np.nan)
            row["R²_w"] = m.get("weighted_r2", np.nan)
            row["MAPE_w"] = m.get("weighted_mape", np.nan)
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(TABLES_DIR / "table3_model_comparison_ols_rf_xgb.csv", index=False)

    # ----------------------------
    # Save models & artifacts
    joblib.dump(model_rf, MODELS_DIR / "rf_thermal_intensity_calibrated.joblib")
    joblib.dump(model_ols, MODELS_DIR / "ols_thermal_intensity_calibrated.joblib")
    joblib.dump(model_xgb, MODELS_DIR / "xgboost_thermal_intensity_calibrated.joblib")

    # Save calibration parameters
    pd.DataFrame([{"a": a_cal, "b": b_cal}]).to_csv(MODELS_DIR / "rf_calibration_params.csv", index=False)

    # Feature importance (post-encoding)
    rf_est = model_rf.model
    rf_feat_names = model_rf.preprocessor.get_feature_names_out()
    importance_rf = (
        pd.DataFrame({"feature": rf_feat_names, "importance": rf_est.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_rf.to_csv(TABLES_DIR / "feature_importance_rf.csv", index=False)

    importance_xgb = (
        pd.DataFrame({"feature": model_xgb.feature_names_, "importance": model_xgb.model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_xgb.to_csv(TABLES_DIR / "feature_importance_xgboost.csv", index=False)

    logger.info("=" * 60)
    logger.info("Modeling complete. Saved models, tables, figures, and calibration outputs.")
    logger.info("=" * 60)

    return {
        "model_rf": model_rf,
        "model_ols": model_ols,
        "model_xgb": model_xgb,

        "rf_train_metrics": rf_train,
        "rf_val_metrics": rf_val,
        "rf_test_metrics": rf_test,

        "ols_train_metrics": ols_train,
        "ols_val_metrics": ols_val,
        "ols_test_metrics": ols_test,

        "xgb_train_metrics": xgb_train,
        "xgb_val_metrics": xgb_val,
        "xgb_test_metrics": xgb_test,

        "cv_results": cv_results,
        "subgroup_metrics": subgroup_metrics,

        "feature_importance_rf": importance_rf,
        "feature_importance_xgb": importance_xgb,

        "model_comparison": comparison_df,
        "model_performance_table": model_perf_df,

        "X_test": X_test,
        "y_test": y_test,
        "y_pred_rf_uncal": ypred_test_rf_uncal,
        "y_pred_rf_cal": ypred_test_rf_cal,
        "calibration_params": (a_cal, b_cal),
        "error_breakdown": df_error_breakdown,
    }


if __name__ == "__main__":
    # choose transform = "none" | "log1p" | "yeo"
    results = run_modeling_pipeline(target_transform="yeo")

    print("\n" + "=" * 60)
    print("MODEL RESULTS SUMMARY")
    print("=" * 60)
    print("\nTest Set Performance (RF main model):")
    rf_test = results['rf_test_metrics']
    print(f"  R²:   {rf_test['r2']:.4f}")
    print(f"  RMSE: {rf_test['rmse']:.4f}")
    print(f"  MAE:  {rf_test['mae']:.4f}")

    print("\nCalibration summary (RF, fitted on Val):")
    print(pd.read_csv(TABLES_DIR / "calibration_summary_rf.csv").to_string(index=False))

    print("\nTop 10 Important Features (RF):")
    print(results["feature_importance_rf"].head(10).to_string(index=False))

    print("\nError breakdown saved to:", TABLES_DIR / "error_by_decile_climate_hdd_rf.csv")
