# app.py
"""
A4M — Streamlit single-file app
- Full EDA + extended model diagnostics
- Dark theme and color schema preserved from user's reference
Run:
    streamlit run app.py
"""
import io, os, time, zipfile, tempfile, subprocess, sys, warnings
from typing import Dict, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import linregress, f_oneway
import xgboost as xgb
import lightgbm as lgb
import joblib

from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
                             accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix, balanced_accuracy_score, f1_score,
                             brier_score_loss, average_precision_score, classification_report)
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# Optional VIF (statsmodels)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    _HAS_STATS_MODELS = True
except Exception:
    _HAS_STATS_MODELS = False

np.random.seed(42)

# -------------------------
# Quiet context manager to suppress stdout/stderr while training
# -------------------------
class HiddenPrints:
    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stdout.close(); sys.stderr.close()
        except Exception:
            pass
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

# -------------------------
# Page + plotting theme
# -------------------------
st.set_page_config(page_title="A4M — Δf & ΔG: predict formation energy & stability", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .stApp { background: #000000; color: #E6EDF3; }
  .sidebar .sidebar-content { background: rgba(255,255,255,0.03); }
  .stButton>button { background-color:#0f766e; border-radius:10px; color:white; }
  .stDownloadButton>button { background-color:#6d28d9; border-radius:10px; color:white; }
  h1, h2, h3 { color: #E6EDF3; }
</style>
""", unsafe_allow_html=True)

sns.set_style("darkgrid")
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "text.color": "white",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "legend.frameon": True,
    "legend.edgecolor": "white",
    "legend.facecolor": "black",
    "legend.labelcolor": "white",
    "legend.loc": "best",
    "figure.dpi": 150
})

sequential_palette = "viridis"
categorical_palette = "bright"

# -------------------------
# Utilities & feature engineering
# -------------------------
def safe_cos_deg(x): return np.cos(np.deg2rad(x))
def safe_sin_deg(x): return np.sin(np.deg2rad(x))

def detect_gpu() -> bool:
    try:
        res = subprocess.run(['nvidia-smi','-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False

def predict_with_best_ntree(bst: xgb.Booster, dmat: xgb.DMatrix):
    best_ntree_limit = getattr(bst, "best_ntree_limit", None)
    best_iteration = getattr(bst, "best_iteration", None)
    ntree = best_ntree_limit if best_ntree_limit is not None else (best_iteration + 1 if best_iteration is not None else None)
    if ntree is not None:
        try:
            return bst.predict(dmat, iteration_range=(0, ntree))
        except TypeError:
            try:
                return bst.predict(dmat, ntree_limit=ntree)
            except TypeError:
                return bst.predict(dmat)
    return bst.predict(dmat)

def safe_mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

@st.cache_data(show_spinner=False)
def load_csv_buffer(buf) -> pd.DataFrame:
    return pd.read_csv(buf)

@st.cache_data(show_spinner=False)
def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    expected_cols = ['lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang',
                     'lattice_angle_alpha_degree','lattice_angle_beta_degree','lattice_angle_gamma_degree',
                     'number_of_total_atoms','percent_atom_al','percent_atom_in',
                     'percent_atom_ga']
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0.0
    try:
        cos_a = safe_cos_deg(df['lattice_angle_alpha_degree'])
        cos_b = safe_cos_deg(df['lattice_angle_beta_degree'])
        cos_g = safe_cos_deg(df['lattice_angle_gamma_degree'])
        inner = 1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g
        inner = np.clip(inner, 0.0, None)
        df['cell_volume'] = df['lattice_vector_1_ang'] * df['lattice_vector_2_ang'] * df['lattice_vector_3_ang'] * np.sqrt(inner)
    except Exception:
        df['cell_volume'] = 0.0
    df['volume_per_atom'] = df['cell_volume'] / (df.get('number_of_total_atoms', 0) + 1e-9)
    df['al_in_ratio'] = df.get('percent_atom_al', 0) / (df.get('percent_atom_in', 0) + 1e-9)
    df['avg_lattice'] = (df.get('lattice_vector_1_ang', 0) + df.get('lattice_vector_2_ang', 0) + df.get('lattice_vector_3_ang', 0)) / 3
    df['sin_alpha'] = safe_sin_deg(df.get('lattice_angle_alpha_degree', 0))
    df['cos_alpha'] = safe_cos_deg(df.get('lattice_angle_alpha_degree', 0))
    df['angle_product_alpha_beta'] = df.get('lattice_angle_alpha_degree', 0) * df.get('lattice_angle_beta_degree', 0)
    df['angle_product_beta_gamma'] = df.get('lattice_angle_beta_degree', 0) * df.get('lattice_angle_gamma_degree', 0)
    df['angle_product_alpha_gamma'] = df.get('lattice_angle_alpha_degree', 0) * df.get('lattice_angle_gamma_degree', 0)
    df['in_lattice_interaction'] = df.get('percent_atom_in', 0) * df.get('lattice_vector_3_ang', 0)
    df['electroneg_diff_al_in'] = abs(1.61 - 1.78)
    df['electroneg_diff_ga_in'] = abs(1.81 - 1.78)
    df['v1_v2_ratio'] = df.get('lattice_vector_1_ang', 0) / (df.get('lattice_vector_2_ang', 0) + 1e-9)
    df['v2_v3_ratio'] = df.get('lattice_vector_2_ang', 0) / (df.get('lattice_vector_3_ang', 0) + 1e-9)
    df['v1_v3_ratio'] = df.get('lattice_vector_1_ang', 0) / (df.get('lattice_vector_3_ang', 0) + 1e-9)
    return df

@st.cache_data(show_spinner=False)
def remove_outliers(df: pd.DataFrame, col: str = 'formation_energy_ev_natom', z_thresh: float = 3.0) -> pd.DataFrame:
    if col not in df.columns:
        return df.reset_index(drop=True)
    try:
        z = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
        keep = z <= z_thresh
        return df[keep].reset_index(drop=True)
    except Exception:
        return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def prepare_features(df: pd.DataFrame):
    """
    Safe preparation:
      - Ensure spacegroup exists and is str
      - OneHot encode spacegroup
      - Select numeric columns, coerce to numeric (safe), fillna(0)
      - Scale numeric cols with RobustScaler
      - Build sparse feature matrix
    """
    df_ml = df.drop(columns=['id'], errors='ignore').copy()
    try:
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')

    if 'spacegroup' not in df_ml.columns:
        df_ml['spacegroup'] = 'sg_0'
    df_ml['spacegroup'] = df_ml['spacegroup'].astype(str)

    exclude = {'spacegroup', 'formation_energy_ev_natom', 'bandgap_energy_ev'}
    candidate_cols = [c for c in df_ml.columns if c not in exclude]

    numeric_df = df_ml[candidate_cols].copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
    numeric_df = numeric_df.fillna(0.0)

    space_encoded = encoder.fit_transform(df_ml[['spacegroup']])
    if not sparse.issparse(space_encoded):
        space_encoded = sparse.csr_matrix(space_encoded)

    scaler = RobustScaler()
    numeric_scaled = scaler.fit_transform(numeric_df.values)
    numeric_matrix = sparse.csr_matrix(numeric_scaled)

    X_sparse = sparse.hstack([numeric_matrix, space_encoded], format='csr')

    try:
        sg_names = encoder.get_feature_names_out(['spacegroup']).tolist()
    except Exception:
        sg_names = [f"spacegroup_{i}" for i in range(space_encoded.shape[1])]

    feature_names = list(numeric_df.columns) + sg_names

    y_reg = df_ml['formation_energy_ev_natom'].fillna(0.0).values.astype(float)
    median_fe = np.median(y_reg)
    y_clf = (y_reg < median_fe).astype(int)
    y_reg_log = np.log1p(np.maximum(y_reg, 0.0))

    return X_sparse, feature_names, y_reg_log, y_clf, encoder, scaler, list(numeric_df.columns)

# -------------------------
# hyperparams + robust search wrapper
# -------------------------
param_dist_common = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.01, 0.03, 0.05, 0.08],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0],
    'reg_alpha': [0.0, 0.1, 0.5]
}

def robust_random_search(estimator, estimator_name: str, param_distributions, X, y, n_iter, cv, scoring, random_state, n_jobs):
    rs = RandomizedSearchCV(estimator, param_distributions=param_distributions, n_iter=n_iter, cv=cv,
                            scoring=scoring, random_state=random_state, n_jobs=n_jobs, verbose=0)
    try:
        with HiddenPrints():
            rs.fit(X, y)
        return rs, None
    except Exception:
        try:
            bp = estimator.get_params()
            if bp.get('tree_method', None) == 'gpu_hist':
                bp['tree_method'] = 'hist'
            if bp.get('predictor', None) == 'gpu_predictor':
                bp['predictor'] = 'auto'
            EstClass = estimator.__class__
            fallback_est = EstClass(**{k: v for k, v in bp.items() if k in EstClass().get_params().keys()})
        except Exception:
            if isinstance(estimator, xgb.XGBRegressor):
                fallback_est = xgb.XGBRegressor(tree_method='hist', predictor='auto', verbosity=0, random_state=random_state, n_jobs=n_jobs)
            elif isinstance(estimator, xgb.XGBClassifier):
                fallback_est = xgb.XGBClassifier(tree_method='hist', predictor='auto', verbosity=0, random_state=random_state, n_jobs=n_jobs, use_label_encoder=False)
            else:
                fallback_est = estimator
        rs2 = RandomizedSearchCV(fallback_est, param_distributions=param_distributions, n_iter=n_iter, cv=cv,
                                 scoring=scoring, random_state=random_state, n_jobs=n_jobs, verbose=0)
        with HiddenPrints():
            rs2.fit(X, y)
        return rs2, f"{estimator_name}: retried with CPU-safe params."

# -------------------------
# training pipeline
# -------------------------
@st.cache_resource(show_spinner=False)
def train_pipeline(_X_sparse, feature_names, y_reg_log, y_clf, quick_mode=True, random_state=42) -> Dict[str, Any]:
    X_sparse = _X_sparse
    use_gpu = detect_gpu()
    tree_method = 'gpu_hist' if use_gpu else 'hist'
    predictor = 'gpu_predictor' if use_gpu else 'auto'
    n_jobs = 1

    X_train_full_reg, X_test_reg, y_train_full_reg, y_test_reg = train_test_split(X_sparse, y_reg_log, test_size=0.20, random_state=random_state)
    X_train_full_clf, X_test_clf, y_train_full_clf, y_test_clf = train_test_split(X_sparse, y_clf, test_size=0.20, random_state=random_state, stratify=y_clf)
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_full_reg, y_train_full_reg, test_size=0.20, random_state=random_state)
    X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(X_train_full_clf, y_train_full_clf, test_size=0.20, random_state=random_state, stratify=y_train_full_clf)

    n_iter_reg = 8 if quick_mode else 30
    cv_reg = 2 if quick_mode else 3

    base_reg = xgb.XGBRegressor(objective='reg:squarederror', tree_method=tree_method, predictor=predictor, verbosity=0, random_state=random_state, n_jobs=n_jobs)
    rand_reg, msg_reg = robust_random_search(base_reg, "XGBRegressor", param_dist_common, X_train_reg, y_train_reg, n_iter_reg, cv_reg, 'r2', random_state, n_jobs)
    if msg_reg:
        print(msg_reg)

    best_reg_params = rand_reg.best_params_.copy()
    n_estimators_reg = int(best_reg_params.pop('n_estimators', 300))
    params_xgb_reg = best_reg_params.copy()
    params_xgb_reg.update({'objective':'reg:squarederror','verbosity':0,'seed':random_state})
    if 'tree_method' in rand_reg.best_params_:
        params_xgb_reg['tree_method'] = rand_reg.best_params_.get('tree_method', 'hist')
    else:
        params_xgb_reg['tree_method'] = 'hist'

    dtrain = xgb.DMatrix(X_train_reg, label=y_train_reg, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_reg, label=y_val_reg, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test_reg, label=y_test_reg, feature_names=feature_names)

    with HiddenPrints():
        bst_reg = xgb.train(params_xgb_reg, dtrain, num_boost_round=n_estimators_reg, evals=[(dtrain,'train'),(dval,'valid')], early_stopping_rounds=20, verbose_eval=False)

    lgb_reg_args = {
        'learning_rate': params_xgb_reg.get('learning_rate', 0.05),
        'num_leaves': 31 if params_xgb_reg.get('max_depth',5) >=5 else 15,
        'feature_fraction': params_xgb_reg.get('colsample_bytree', 0.8),
        'bagging_fraction': params_xgb_reg.get('subsample', 0.8),
        'lambda_l1': params_xgb_reg.get('reg_alpha', 0.0),
        'lambda_l2': params_xgb_reg.get('reg_lambda', 1.0),
        'verbosity': -1,
        'random_state': random_state,
        'n_jobs': n_jobs
    }
    lgb_reg = lgb.LGBMRegressor(n_estimators=n_estimators_reg, **lgb_reg_args)
    try:
        with HiddenPrints():
            callbacks = [lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
            lgb_reg.fit(X_train_reg, y_train_reg, eval_set=[(X_val_reg, y_val_reg)], eval_metric='rmse', callbacks=callbacks)
    except Exception:
        with HiddenPrints():
            lgb_reg.fit(X_train_reg, y_train_reg)

    # regression preds and metrics
    y_test_pred_log_xgb = predict_with_best_ntree(bst_reg, dtest)
    y_test_pred_log_lgb = lgb_reg.predict(X_test_reg, num_iteration=getattr(lgb_reg, "best_iteration_", None))
    y_test_pred_log_ens = (y_test_pred_log_xgb + y_test_pred_log_lgb) / 2.0
    y_test_pred_ens = np.expm1(y_test_pred_log_ens)
    y_test_orig = np.expm1(y_test_reg)

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_ens))
    mae = mean_absolute_error(y_test_orig, y_test_pred_ens)
    r2 = r2_score(y_test_orig, y_test_pred_ens)
    evs = explained_variance_score(y_test_orig, y_test_pred_ens)
    mape = safe_mape(y_test_orig, y_test_pred_ens)

    # classification
    n_iter_clf = 8 if quick_mode else 30
    cv_clf = 2 if quick_mode else 3

    base_clf = xgb.XGBClassifier(objective='binary:logistic', tree_method=tree_method, predictor=predictor, verbosity=0, random_state=random_state, n_jobs=n_jobs, use_label_encoder=False)
    rand_clf, msg_clf = robust_random_search(base_clf, "XGBClassifier", param_dist_common, X_train_clf, y_train_clf, n_iter_clf, cv_clf, 'f1', random_state, n_jobs)
    if msg_clf:
        print(msg_clf)

    best_clf_params = rand_clf.best_params_.copy()
    n_estimators_clf = int(best_clf_params.pop('n_estimators', 300))
    params_xgb_clf = best_clf_params.copy()
    params_xgb_clf.update({'objective':'binary:logistic','eval_metric':'auc','verbosity':0,'seed':random_state})
    if 'tree_method' in rand_clf.best_params_:
        params_xgb_clf['tree_method'] = rand_clf.best_params_.get('tree_method', 'hist')
    else:
        params_xgb_clf['tree_method'] = 'hist'

    dtrain_clf = xgb.DMatrix(X_train_clf, label=y_train_clf, feature_names=feature_names)
    dval_clf = xgb.DMatrix(X_val_clf, label=y_val_clf, feature_names=feature_names)
    dtest_clf = xgb.DMatrix(X_test_clf, label=y_test_clf, feature_names=feature_names)

    with HiddenPrints():
        bst_clf = xgb.train(params_xgb_clf, dtrain_clf, num_boost_round=n_estimators_clf, evals=[(dtrain_clf,'train'),(dval_clf,'valid')], early_stopping_rounds=20, verbose_eval=False)

    lgb_clf_args = {
        'learning_rate': params_xgb_clf.get('learning_rate', 0.05),
        'num_leaves': 31 if params_xgb_clf.get('max_depth',5) >=5 else 15,
        'feature_fraction': params_xgb_clf.get('colsample_bytree', 0.8),
        'bagging_fraction': params_xgb_clf.get('subsample', 0.8),
        'lambda_l1': params_xgb_clf.get('reg_alpha', 0.0),
        'lambda_l2': params_xgb_clf.get('reg_lambda', 1.0),
        'verbosity': -1,
        'random_state': random_state,
        'n_jobs': n_jobs
    }
    lgb_clf = lgb.LGBMClassifier(n_estimators=n_estimators_clf, **lgb_clf_args)
    try:
        with HiddenPrints():
            callbacks_clf = [lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
            lgb_clf.fit(X_train_clf, y_train_clf, eval_set=[(X_val_clf, y_val_clf)], eval_metric='auc', callbacks=callbacks_clf)
    except Exception:
        with HiddenPrints():
            lgb_clf.fit(X_train_clf, y_train_clf)

    # ensemble probabilities + metrics
    y_proba_xgb = predict_with_best_ntree(bst_clf, dtest_clf)
    y_proba_lgb = lgb_clf.predict_proba(X_test_clf, num_iteration=getattr(lgb_clf, "best_iteration_", None))[:,1]
    y_proba_ens = (y_proba_xgb + y_proba_lgb) / 2.0

    val_proba_xgb = predict_with_best_ntree(bst_clf, xgb.DMatrix(X_val_clf, feature_names=feature_names))
    val_proba_lgb = lgb_clf.predict_proba(X_val_clf, num_iteration=getattr(lgb_clf,"best_iteration_", None))[:,1]
    val_proba_ens = (val_proba_xgb + val_proba_lgb) / 2.0

    best_thr = 0.5
    best_f1 = -1
    for thr in np.linspace(0.01, 0.99, 99):
        yv = (val_proba_ens >= thr).astype(int)
        f1v = f1_score(y_val_clf, yv, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_thr = thr
    optimal_threshold = best_thr

    y_pred_ens = (y_proba_ens >= optimal_threshold).astype(int)

    acc = accuracy_score(y_test_clf, y_pred_ens)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_clf, y_pred_ens, average='binary', zero_division=0)
    auc = roc_auc_score(y_test_clf, y_proba_ens)
    pr_auc = average_precision_score(y_test_clf, y_proba_ens)
    brier = brier_score_loss(y_test_clf, y_proba_ens)
    bal_acc = balanced_accuracy_score(y_test_clf, y_pred_ens)

    # CV quick checks
    cv_info = {}
    try:
        reg_cv_model = xgb.XGBRegressor(n_estimators=n_estimators_reg, **{k:v for k,v in rand_reg.best_params_.items() if k!='n_estimators'}, tree_method=params_xgb_reg.get('tree_method','hist'), predictor='auto', verbosity=0, random_state=random_state)
        kf = KFold(n_splits=3 if quick_mode else 5, shuffle=True, random_state=random_state)
        with HiddenPrints():
            r2_cv = cross_val_score(reg_cv_model, X_sparse, y_reg_log, cv=kf, scoring='r2', n_jobs=1)
        cv_info['reg_r2_mean'] = float(r2_cv.mean()); cv_info['reg_r2_std'] = float(r2_cv.std())
    except Exception as e:
        cv_info['reg_cv_error'] = str(e)
    try:
        clf_cv_model = xgb.XGBClassifier(n_estimators=n_estimators_clf, **{k:v for k,v in rand_clf.best_params_.items() if k!='n_estimators'}, tree_method=params_xgb_clf.get('tree_method','hist'), predictor='auto', verbosity=0, random_state=random_state, use_label_encoder=False)
        skf = StratifiedKFold(n_splits=3 if quick_mode else 5, shuffle=True, random_state=random_state)
        with HiddenPrints():
            f1_cv = cross_val_score(clf_cv_model, X_sparse, y_clf, cv=skf, scoring='f1', n_jobs=1)
            auc_cv = cross_val_score(clf_cv_model, X_sparse, y_clf, cv=skf, scoring='roc_auc', n_jobs=1)
        cv_info['clf_f1_mean'] = float(f1_cv.mean()); cv_info['clf_auc_mean'] = float(auc_cv.mean())
    except Exception as e:
        cv_info['clf_cv_error'] = str(e)

    out_reg = pd.DataFrame({'Actual_formation': y_test_orig, 'Predicted_formation': y_test_pred_ens})
    out_clf = pd.DataFrame({'Actual_stable': y_test_clf, 'Predicted_stable': y_pred_ens, 'Proba_stable': y_proba_ens})

    artifacts = {'bst_reg': bst_reg, 'lgb_reg': lgb_reg, 'bst_clf': bst_clf, 'lgb_clf': lgb_clf, 'out_reg': out_reg, 'out_clf': out_clf}
    metrics = {'reg_rmse': float(rmse), 'reg_mae': float(mae), 'reg_r2': float(r2), 'reg_evs': float(evs), 'reg_mape': float(mape),
               'clf_acc': float(acc), 'clf_precision': float(precision), 'clf_recall': float(recall), 'clf_f1': float(f1), 'clf_auc': float(auc), 'clf_pr_auc': float(pr_auc), 'clf_brier': float(brier), 'clf_bal_acc': float(bal_acc),
               'optimal_threshold': float(optimal_threshold)}
    return {'artifacts': artifacts, 'metrics': metrics, 'cv_info': cv_info, 'feature_names': feature_names}

# -------------------------
# Streamlit UI
# -------------------------
st.title("A4M — Δf & ΔG: predict formation energy and thermodynamic stability")
st.write("Upload `train.csv`. QUICK mode reduces randomized-search work for demos.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload train.csv", type=['csv'])
    quick_mode = st.checkbox("QUICK mode (faster demo)", value=True)
    run_button = st.button("Run pipeline")
    st.markdown("---")

if uploaded_file is None:
    st.warning("Please upload train.csv (small sample works).")
    st.stop()

df = load_csv_buffer(uploaded_file)
st.subheader("Dataset preview")
st.dataframe(df.head())
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0]); c2.metric("Cols", df.shape[1]); c3.metric("Missing", int(df.isna().sum().sum()))

with st.spinner("Deriving features for EDA..."):
    df_fe = derive_features(df)
    df_fe = remove_outliers(df_fe, col='formation_energy_ev_natom', z_thresh=3.0)
    st.success(f"Derived features, shape: {df_fe.shape}")

# -------------------------
# Full EDA & Statistical Insights (explicit; no placeholder)
# -------------------------
with st.expander("Full EDA & Statistical Insights (click to expand)"):
    def add_explanation(text, fig):
        fig.text(0.1, -0.05, text, wrap=True, horizontalalignment='left', fontsize=10, style='italic', color='white')

    # 1) Histogram: Formation energy (teal)
    if 'formation_energy_ev_natom' in df_fe.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df_fe['formation_energy_ev_natom'].dropna(), bins=25, kde=True, color='teal', ax=ax, edgecolor='white', linewidth=0.5)
        ax.set_title('Distribution of Formation Energy per Atom')
        ax.set_xlabel('Formation Energy (eV/atom)'); ax.set_ylabel('Frequency')
        ax.axvline(df_fe['formation_energy_ev_natom'].mean(), color='red', linestyle='--', label=f"Mean: {df_fe['formation_energy_ev_natom'].mean():.3f}")
        ax.axvline(df_fe['formation_energy_ev_natom'].median(), color='orange', linestyle='--', label=f"Median: {df_fe['formation_energy_ev_natom'].median():.3f}")
        ax.legend(); plt.tight_layout(pad=2.0)
        add_explanation("This histogram shows the distribution of formation energies. Mean and median lines highlight central tendency.", fig)
        st.pyplot(fig)

    # 2) Histogram: Bandgap energy (purple)
    if 'bandgap_energy_ev' in df_fe.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df_fe['bandgap_energy_ev'].dropna(), bins=25, kde=True, color='purple', ax=ax, edgecolor='white', linewidth=0.5)
        ax.set_title('Distribution of Bandgap Energy')
        ax.set_xlabel('Bandgap Energy (eV)'); ax.set_ylabel('Frequency')
        ax.axvline(df_fe['bandgap_energy_ev'].mean(), color='red', linestyle='--', label=f"Mean: {df_fe['bandgap_energy_ev'].mean():.3f}")
        ax.axvline(df_fe['bandgap_energy_ev'].median(), color='orange', linestyle='--', label=f"Median: {df_fe['bandgap_energy_ev'].median():.3f}")
        ax.legend(); plt.tight_layout(pad=2.0)
        add_explanation("Bandgap distribution; KDE helps identify peaks.", fig)
        st.pyplot(fig)

    # 3) Scatter: Formation vs Bandgap with trendline
    if {'bandgap_energy_ev','formation_energy_ev_natom'}.issubset(df_fe.columns):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x='bandgap_energy_ev', y='formation_energy_ev_natom', data=df_fe, alpha=0.7, s=40, edgecolor='white', linewidth=0.5, ax=ax)
        try:
            slope, intercept, r_value, _, _ = linregress(df_fe['bandgap_energy_ev'].fillna(0), df_fe['formation_energy_ev_natom'].fillna(0))
            ax.plot(df_fe['bandgap_energy_ev'], intercept + slope * df_fe['bandgap_energy_ev'], color='red', linestyle='--', label=f'Trend (r={r_value:.2f})')
            ax.legend()
        except Exception:
            pass
        ax.set_title('Formation Energy vs Bandgap Energy'); ax.set_xlabel('Bandgap Energy (eV)'); ax.set_ylabel('Formation Energy (eV/atom)')
        plt.tight_layout(pad=2.0); add_explanation("Scatter + trendline shows relationship between bandgap and formation energy.", fig); st.pyplot(fig)

    # 4) Correlation heatmap (coolwarm)
    try:
        numeric = df_fe.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')
        corr_matrix = numeric.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, mask=mask, cbar_kws={'shrink':0.8}, ax=ax, annot_kws={"color":"white"})
        ax.set_title('Correlation Matrix Heatmap'); plt.tight_layout(pad=2.0)
        add_explanation("Correlation matrix (upper triangle masked).", fig)
        st.pyplot(fig)
    except Exception:
        pass

    # 5) Boxplot: Formation by spacegroup + swarm
    if {'spacegroup','formation_energy_ev_natom'}.issubset(df_fe.columns):
        try:
            fig, ax = plt.subplots(figsize=(10,5))
            sns.boxplot(x='spacegroup', y='formation_energy_ev_natom', data=df_fe, palette=categorical_palette, width=0.5, ax=ax, linewidth=1.0)
            sns.swarmplot(x='spacegroup', y='formation_energy_ev_natom', data=df_fe, color='white', alpha=0.4, size=3, ax=ax)
            ax.set_title('Formation Energy by Spacegroup')
            ax.set_xlabel('Spacegroup'); ax.set_ylabel('Formation Energy (eV/atom)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            plt.tight_layout(pad=2.0)
            add_explanation("Boxplot + swarm shows distribution across spacegroups.", fig)
            st.pyplot(fig)
        except Exception:
            pass

    # 6) Scatter: %In vs Bandgap with trendline
    if {'percent_atom_in','bandgap_energy_ev'}.issubset(df_fe.columns):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x='percent_atom_in', y='bandgap_energy_ev', data=df_fe, alpha=0.7, s=40, edgecolor='white', linewidth=0.5, ax=ax)
        try:
            slope, intercept, r_value, _, _ = linregress(df_fe['percent_atom_in'].fillna(0), df_fe['bandgap_energy_ev'].fillna(0))
            ax.plot(df_fe['percent_atom_in'], intercept + slope * df_fe['percent_atom_in'], color='red', linestyle='--', label=f'Trend (r={r_value:.2f})')
            ax.legend()
        except Exception:
            pass
        ax.set_title('%In vs Bandgap Energy'); plt.tight_layout(pad=2.0); add_explanation("Higher %In often correlates with lower bandgap.", fig); st.pyplot(fig)

    # 7) Countplot: number_of_total_atoms with percentages
    if 'number_of_total_atoms' in df_fe.columns:
        try:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.countplot(x='number_of_total_atoms', data=df_fe, palette=categorical_palette, edgecolor='white', linewidth=0.5, ax=ax)
            total = len(df_fe)
            for p in ax.patches:
                percentage = f'{100 * p.get_height() / total:.1f}%'
                ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0,9), textcoords='offset points', fontsize=10, color='white')
            ax.set_title('Distribution of Total Atoms per Structure'); plt.tight_layout(pad=2.0)
            add_explanation("Counts annotated with percentages.", fig); st.pyplot(fig)
        except Exception:
            pass

    # 8) Pairplot: lattice vectors vs formation energy (may be slow)
    if {'lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang','formation_energy_ev_natom','spacegroup'}.issubset(df_fe.columns):
        try:
            st.markdown("**Pairplot (lattice vectors vs formation energy colored by spacegroup)** — may be slow for large datasets.")
            pair_cols = ['lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang','formation_energy_ev_natom','spacegroup']
            g = sns.pairplot(df_fe[pair_cols], hue='spacegroup', palette=categorical_palette, diag_kind='kde',
                             plot_kws={'alpha':0.7,'s':30,'edgecolor':'white','linewidth':0.5}, height=2.0)
            g.fig.suptitle('Pairplot: Lattice Vectors vs Formation Energy (Colored by Spacegroup)', y=1.02)
            st.pyplot(g.fig)
        except Exception:
            pass

    # 9) Ternary composition plot (discrete high-contrast palette)
    if {'percent_atom_al','percent_atom_ga','percent_atom_in','formation_energy_ev_natom'}.issubset(df_fe.columns):
        try:
            df_fe['_fe_bin'] = pd.qcut(df_fe['formation_energy_ev_natom'].fillna(df_fe['formation_energy_ev_natom'].median()), q=5, labels=[f"q{i+1}" for i in range(5)], duplicates='drop')
            color_seq = px.colors.qualitative.Bold
            fig = px.scatter_ternary(df_fe, a="percent_atom_al", b="percent_atom_ga", c="percent_atom_in",
                                     color="_fe_bin", color_discrete_sequence=color_seq,
                                     hover_data=['formation_energy_ev_natom'], size_max=6,
                                     title="Ternary Composition Plot (binned formation energy -> discrete colors)")
            fig.update_layout(width=700, height=520, title_font_size=14, paper_bgcolor="black", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # 10) Violin plot: Bandgap by total atoms
    if {'number_of_total_atoms','bandgap_energy_ev'}.issubset(df_fe.columns):
        try:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.violinplot(x='number_of_total_atoms', y='bandgap_energy_ev', data=df_fe, palette=categorical_palette, inner='quartile', ax=ax, linewidth=1.0)
            ax.set_title('Violin Plot: Bandgap Energy by Total Atoms'); plt.tight_layout(pad=2.0)
            add_explanation("Violin plot shows distribution density by atom count.", fig); st.pyplot(fig)
        except Exception:
            pass

    # 11) PCA: percent atoms + lattice vectors
    try:
        pca_cols = ['percent_atom_al','percent_atom_ga','percent_atom_in','lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang']
        pca_cols = [c for c in pca_cols if c in df_fe.columns]
        if len(pca_cols) >= 2:
            scaler = StandardScaler()
            pca_features = scaler.fit_transform(df_fe[pca_cols].fillna(0))
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(pca_features)
            df_fe['PCA1'] = pca_result[:,0]; df_fe['PCA2'] = pca_result[:,1]
            fig, ax = plt.subplots(figsize=(8,5))
            hue_col = 'spacegroup' if 'spacegroup' in df_fe.columns else None
            sns.scatterplot(x='PCA1', y='PCA2', hue=hue_col, data=df_fe, palette=categorical_palette, alpha=0.7, s=40, edgecolor='white', linewidth=0.5, ax=ax)
            ax.set_title('PCA: Reduced Dimensions (Colored by Spacegroup)'); plt.tight_layout(pad=2.0)
            add_explanation("PCA reduces features to 2D; clusters show grouping by spacegroup.", fig); st.pyplot(fig)
            st.write("Explained variance ratio:", pca.explained_variance_ratio_.round(3).tolist())
    except Exception:
        pass

    # 12) Advanced stats: skew/kurtosis, cell volume, correlations, VIF, ANOVA
    st.markdown("**Advanced stats & diagnostics**")
    try:
        if 'formation_energy_ev_natom' in df_fe.columns:
            st.write("Skewness (formation energy):", float(df_fe['formation_energy_ev_natom'].skew()))
            st.write("Kurtosis (formation energy):", float(df_fe['formation_energy_ev_natom'].kurtosis()))
        if 'bandgap_energy_ev' in df_fe.columns:
            st.write("Skewness (bandgap):", float(df_fe['bandgap_energy_ev'].skew()))
            st.write("Kurtosis (bandgap):", float(df_fe['bandgap_energy_ev'].kurtosis()))
    except Exception:
        pass

    try:
        a = df_fe.get('lattice_vector_1_ang', pd.Series(dtype=float))
        b = df_fe.get('lattice_vector_2_ang', pd.Series(dtype=float))
        c = df_fe.get('lattice_vector_3_ang', pd.Series(dtype=float))
        alpha = np.deg2rad(df_fe.get('lattice_angle_alpha_degree', pd.Series(dtype=float)))
        beta = np.deg2rad(df_fe.get('lattice_angle_beta_degree', pd.Series(dtype=float)))
        gamma = np.deg2rad(df_fe.get('lattice_angle_gamma_degree', pd.Series(dtype=float)))
        df_fe['cell_volume'] = a * b * c * np.sqrt(np.clip(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma), 0.0, None))
        st.write("Cell volume statistics:")
        st.write(df_fe['cell_volume'].describe().to_frame().T)
        if 'formation_energy_ev_natom' in df_fe.columns:
            st.write("Correlation of cell_volume with formation_energy:", float(df_fe['cell_volume'].corr(df_fe['formation_energy_ev_natom'])))
        if 'bandgap_energy_ev' in df_fe.columns:
            st.write("Correlation of cell_volume with bandgap:", float(df_fe['cell_volume'].corr(df_fe['bandgap_energy_ev'])))
    except Exception:
        pass

    try:
        vif_cols = ['percent_atom_al','percent_atom_ga','percent_atom_in','lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang']
        vif_cols = [c for c in vif_cols if c in df_fe.columns]
        if _HAS_STATS_MODELS and len(vif_cols) >= 2:
            features = df_fe[vif_cols].fillna(0)
            vif_data = pd.DataFrame({"feature": vif_cols, "VIF": [variance_inflation_factor(features.values, i) for i in range(len(vif_cols))]})
            st.write("VIF (multicollinearity):")
            st.dataframe(vif_data)
    except Exception:
        pass

    try:
        if 'spacegroup' in df_fe.columns and 'formation_energy_ev_natom' in df_fe.columns:
            groups = [grp['formation_energy_ev_natom'].dropna().values for _, grp in df_fe.groupby('spacegroup')]
            if len(groups) >= 2:
                anova_res = f_oneway(*groups)
                st.write("ANOVA formation_energy_ev_natom by spacegroup: F =", float(anova_res.statistic), ", p =", float(anova_res.pvalue))
    except Exception:
        pass

    # 13) Feature importances via RandomForest for formation energy
    try:
        fi_cols = ['spacegroup','number_of_total_atoms','percent_atom_al','percent_atom_ga','percent_atom_in',
                   'lattice_vector_1_ang','lattice_vector_2_ang','lattice_vector_3_ang',
                   'lattice_angle_alpha_degree','lattice_angle_beta_degree','lattice_angle_gamma_degree']
        fi_cols = [c for c in fi_cols if c in df_fe.columns]
        if 'formation_energy_ev_natom' in df_fe.columns and len(fi_cols) >= 2:
            # encode spacegroup if present for RF
            X_fi = df_fe[fi_cols].copy()
            if 'spacegroup' in X_fi.columns:
                X_fi = pd.get_dummies(X_fi, columns=['spacegroup'], drop_first=True)
            X_fi = X_fi.fillna(0)
            y_fi = df_fe['formation_energy_ev_natom'].fillna(0)
            rf = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=1)
            rf.fit(X_fi, y_fi)
            importances = pd.DataFrame({'Feature': X_fi.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x='Importance', y='Feature', data=importances.head(25), palette=sequential_palette, ax=ax, edgecolor='white', linewidth=0.5)
            ax.set_title('RandomForest Feature Importances (formation energy)')
            plt.tight_layout(pad=2.0); st.pyplot(fig)
            st.dataframe(importances.head(50).reset_index(drop=True))
    except Exception:
        pass

# -------------------------
# Run training and show diagnostics
# -------------------------
if run_button:
    t0 = time.time()
    st.info("Running pipeline — this can take some time (QUICK mode speeds this up).")

    with st.spinner("Preparing features for training..."):
        X_sparse, feature_names, y_reg_log, y_clf, encoder, scaler, numeric_cols = prepare_features(df_fe)
        st.success("Features ready")

    with st.spinner("Training models..."):
        result = train_pipeline(X_sparse, feature_names, y_reg_log, y_clf, quick_mode=quick_mode)

    metrics = result['metrics']
    st.subheader("Summary metrics")
    st.metric("Regression RMSE", f"{metrics['reg_rmse']:.6f}"); st.metric("Regression R²", f"{metrics['reg_r2']:.4f}")
    st.write(f"Regression MAE: {metrics['reg_mae']:.6f} — MAPE: {metrics['reg_mape']:.3f}%")
    st.metric("Classification F1", f"{metrics['clf_f1']:.4f}"); st.metric("Classification AUC", f"{metrics['clf_auc']:.4f}")
    st.write(f"Accuracy: {metrics['clf_acc']:.4f}, Brier: {metrics['clf_brier']:.5f}, Threshold: {metrics['optimal_threshold']:.3f}")

    if result.get('cv_info'):
        st.write("CV checks:", result['cv_info'])

    # ---------- Regression diagnostics ----------
    st.subheader("Regression diagnostics (XGB + LGB ensemble)")
    try:
        out_reg = result['artifacts']['out_reg']
        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x='Actual_formation', y='Predicted_formation', data=out_reg, edgecolor='white', s=40, ax=ax)
        ax.plot([out_reg['Actual_formation'].min(), out_reg['Actual_formation'].max()],
                 [out_reg['Actual_formation'].min(), out_reg['Actual_formation'].max()], color='red', linestyle='--')
        ax.set_title('Actual vs Predicted (Regression)'); plt.tight_layout(pad=1.0); st.pyplot(fig)

        # Residuals histogram
        res = out_reg['Actual_formation'] - out_reg['Predicted_formation']
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(res, bins=30, kde=True, ax=ax, edgecolor='white'); ax.set_title('Residuals (Actual - Predicted)'); plt.tight_layout(pad=1.0); st.pyplot(fig)

        # Residuals vs Predicted
        fig, ax = plt.subplots(figsize=(6,3))
        sns.scatterplot(x=out_reg['Predicted_formation'], y=res, edgecolor='white', s=35, ax=ax); ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Predicted formation'); ax.set_ylabel('Residual'); ax.set_title('Residuals vs Predicted'); plt.tight_layout(pad=1.0); st.pyplot(fig)

        # QQ plot
        try:
            import scipy.stats as ss
            fig = plt.figure(figsize=(5,4)); ax = fig.add_subplot(111)
            ss.probplot(res.dropna(), plot=ax); ax.set_title('QQ-plot of Residuals'); plt.tight_layout(pad=1.0); st.pyplot(fig)
        except Exception:
            pass

        # Residuals violin by predicted quantile
        try:
            out_reg['pred_q'] = pd.qcut(out_reg['Predicted_formation'], q=4, labels=False, duplicates='drop')
            fig, ax = plt.subplots(figsize=(6,3))
            sns.violinplot(x='pred_q', y=res, data=out_reg, inner='quartile', ax=ax, palette='viridis')
            ax.set_title('Residuals by Predicted quantile'); ax.set_xlabel('Predicted quantile'); ax.set_ylabel('Residual'); plt.tight_layout(pad=1.0); st.pyplot(fig)
        except Exception:
            pass

        # XGB regressor feature importance (gain)
        try:
            bst_reg = result['artifacts']['bst_reg']
            fmap = bst_reg.get_score(importance_type='gain')
            items = []
            for k, v in fmap.items():
                if k.startswith('f'):
                    idx = int(k[1:])
                    name = result['feature_names'][idx] if idx < len(result['feature_names']) else k
                else:
                    name = k
                items.append((name, v))
            fi_reg = pd.DataFrame(items, columns=['feature','gain']).sort_values('gain', ascending=False).head(25)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x='gain', y='feature', data=fi_reg, palette=sequential_palette, ax=ax, edgecolor='white')
            ax.set_title('XGBoost Regressor Feature Importance (gain, top 25)'); plt.tight_layout(pad=1.0); st.pyplot(fig)
            st.dataframe(fi_reg.reset_index(drop=True))
        except Exception:
            pass

        # Combined importance (LightGBM + XGB)
        try:
            lgb_reg = result['artifacts'].get('lgb_reg', None)
            if lgb_reg is not None and hasattr(lgb_reg, "feature_importances_"):
                fi_lgb = pd.DataFrame({'feature': result['feature_names'], 'lgb_importance': lgb_reg.feature_importances_[:len(result['feature_names'])]})
                if 'fi_reg' in locals():
                    merged = fi_reg.merge(fi_lgb, on='feature', how='outer').fillna(0)
                else:
                    merged = fi_lgb.sort_values('lgb_importance', ascending=False).head(25)
                merged_sorted = merged.sort_values(by=['lgb_importance'], ascending=True).tail(25)
                fig, ax = plt.subplots(figsize=(8,6))
                merged_sorted.plot.barh(x='feature', y=[col for col in merged_sorted.columns if col!='feature'], ax=ax)
                ax.set_title('Combined importances (LightGBM & XGB)'); plt.tight_layout(pad=1.0); st.pyplot(fig)
        except Exception:
            pass

    except Exception:
        st.write("Regression diagnostics failed to render.")

    # ---------- Classification diagnostics ----------
    st.subheader("Classification diagnostics (XGB + LGB ensemble)")
    try:
        out_clf = result['artifacts']['out_clf']
        y_true = out_clf['Actual_stable'].values; y_proba = out_clf['Proba_stable'].values; y_pred = out_clf['Predicted_stable'].values

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(6,4)); ax.plot(fpr, tpr, label=f"AUC={metrics['clf_auc']:.3f}"); ax.plot([0,1],[0,1], linestyle='--', color='gray'); ax.set_title('ROC Curve'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(); plt.tight_layout(pad=1.0); st.pyplot(fig)

        # Precision-Recall
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(6,4)); ax.plot(recall_vals, precision_vals); ax.set_title('Precision-Recall Curve'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); plt.tight_layout(pad=1.0); st.pyplot(fig)

        # Calibration
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            fig, ax = plt.subplots(figsize=(6,4)); ax.plot(prob_pred, prob_true, marker='o', label='Ensemble'); ax.plot([0,1],[0,1], linestyle='--', color='gray'); ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives'); ax.set_title('Calibration (Reliability diagram)'); ax.legend(); plt.tight_layout(pad=1.0); st.pyplot(fig)
        except Exception:
            pass

        # Probability histograms (teal vs purple)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(y_proba[y_true==1], bins=25, alpha=0.7, label='Positive (stable)', edgecolor='white', color='teal')
        ax.hist(y_proba[y_true==0], bins=25, alpha=0.6, label='Negative (unstable)', edgecolor='white', color='purple')
        ax.set_title('Predicted Probability Distribution'); ax.set_xlabel('Predicted probability'); ax.legend(); plt.tight_layout(pad=1.0); st.pyplot(fig)

        # Threshold sweep (F1/precision/recall)
        thr_range = np.linspace(0.01, 0.99, 99)
        f1s = []; precs = []; recs = []
        for thr in thr_range:
            ythr = (y_proba >= thr).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_true, ythr, average='binary', zero_division=0)
            precs.append(p); recs.append(r); f1s.append(f)
        fig, ax = plt.subplots(figsize=(6,4)); ax.plot(thr_range, f1s, label='F1'); ax.plot(thr_range, precs, label='Precision'); ax.plot(thr_range, recs, label='Recall'); ax.axvline(metrics['optimal_threshold'], linestyle='--', color='gray', label=f"opt thr {metrics['optimal_threshold']:.2f}"); ax.set_xlabel('Threshold'); ax.set_ylabel('Score'); ax.set_title('Threshold tuning (test set)'); ax.legend(); plt.tight_layout(pad=1.0); st.pyplot(fig)

        # Confusion matrices
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax); ax.set_title('Confusion matrix (counts)'); plt.tight_layout(pad=1.0); st.pyplot(fig)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(4,3)); sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis', ax=ax); ax.set_title('Confusion matrix (normalized)'); plt.tight_layout(pad=1.0); st.pyplot(fig)

        # Classification report
        try:
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep).T.round(3)
            st.write("Classification report (test set)"); st.dataframe(rep_df)
        except Exception:
            pass

        # XGBoost classifier feature importance
        try:
            bst_clf = result['artifacts']['bst_clf']
            fmap = bst_clf.get_score(importance_type='gain')
            items = []
            for k, v in fmap.items():
                if k.startswith('f'):
                    idx = int(k[1:])
                    name = result['feature_names'][idx] if idx < len(result['feature_names']) else k
                else:
                    name = k
                items.append((name, v))
            fi_clf = pd.DataFrame(items, columns=['feature','gain']).sort_values('gain', ascending=False).head(25)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x='gain', y='feature', data=fi_clf, palette=sequential_palette, ax=ax, edgecolor='white')
            ax.set_title('XGBoost Classifier Feature Importance (gain, top 25)'); plt.tight_layout(pad=1.0); st.pyplot(fig)
            st.dataframe(fi_clf.reset_index(drop=True))
        except Exception:
            pass

    except Exception:
        st.write("Classification diagnostics failed to render.")

    # artifacts zip
    try:
        tmp = tempfile.mkdtemp(); zip_path = os.path.join(tmp, "artifacts.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr('regression_predictions.csv', result['artifacts']['out_reg'].to_csv(index=False))
            zf.writestr('classification_predictions.csv', result['artifacts']['out_clf'].to_csv(index=False))
            try:
                zf.writestr('xgb_reg.bst', result['artifacts']['bst_reg'].save_raw())
            except Exception:
                p = os.path.join(tmp, 'xgb_reg.bst'); result['artifacts']['bst_reg'].save_model(p); zf.write(p, arcname='xgb_reg.bst')
            try:
                zf.writestr('xgb_clf.bst', result['artifacts']['bst_clf'].save_raw())
            except Exception:
                p = os.path.join(tmp, 'xgb_clf.bst'); result['artifacts']['bst_clf'].save_model(p); zf.write(p, arcname='xgb_clf.bst')
            try:
                lreg = os.path.join(tmp, 'lgb_reg.joblib'); joblib.dump(result['artifacts']['lgb_reg'], lreg, compress=3); zf.write(lreg, arcname='lgb_reg.joblib')
            except Exception: pass
            try:
                lclf = os.path.join(tmp, 'lgb_clf.joblib'); joblib.dump(result['artifacts']['lgb_clf'], lclf, compress=3); zf.write(lclf, arcname='lgb_clf.joblib')
            except Exception: pass
        with open(zip_path, 'rb') as f: st.download_button("Download artifacts (.zip)", data=f, file_name="materials_ml_artifacts.zip")
    except Exception:
        pass

    st.success(f"Finished in {(time.time()-t0):.1f}s")

# Footer: requirements
with st.expander("Deployment & requirements"):
    st.markdown("""
    requirements.txt (suggested):
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    xgboost
    lightgbm
    joblib
    scipy
    plotly
    statsmodels  # optional, for VIF
    ```
    """)
