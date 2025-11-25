from __future__ import annotations
from typing import Optional, Dict, Any
from typing import Optional, List, Literal
from typing import Any, Optional

import numpy as np
import pandas as pd
import pickle
import time
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.multiclass import type_of_target
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_recall_curve, average_precision_score


import plotly.graph_objects as go


import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Literal

# GBM libraries
_HAS_LGB = _HAS_XGB = _HAS_CAT = False
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    pass
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    pass
try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    pass


### GBM Detection
def checkGBM_engine():
    _HAS_LGB = _HAS_XGB = _HAS_CAT = False
    try:
        import lightgbm as lgb
        _HAS_LGB = True
    except Exception:
        pass
    try:
        import xgboost as xgb
        _HAS_XGB = True
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        _HAS_CAT = True
    except Exception:
        pass
    return {
        "_HAS_LGB": _HAS_LGB,
        "_HAS_XGB": _HAS_XGB,
        "_HAS_CAT": _HAS_CAT
    }

### loads datasets
def get_datasets_from_session(use_preprocessed: bool):
    if use_preprocessed:
        return (
            st.session_state._PP_X_train, 
            st.session_state._PP_y_train,
            st.session_state._PP_X_validate, 
            st.session_state._PP_y_validate,
            st.session_state._PP_X_test, 
            st.session_state._PP_y_test,
            st.session_state._PP_SWTR, 
            st.session_state._PP_SWVA, 
            st.session_state._PP_SWTE,
            st.session_state._PP_UE, 
            st.session_state._PP_META
        )
    else:
        return (
            st.session_state._X_train, 
            st.session_state._y_train,
            st.session_state._X_validate, 
            st.session_state._y_validate,
            st.session_state._X_test, 
            st.session_state._y_test,
            None, 
            None, 
            None, 
            None, 
            None
        )

###############
### VERSION ###
###############

def _make_ohe_sparse():
    # Compatible kwargs across sklearn versions
    kwargs = dict(handle_unknown="ignore")
    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        kwargs["sparse_output"] = True
    else:
        kwargs["sparse"] = True
    # Optionally limit category explosion if supported
    if "max_categories" in OneHotEncoder.__init__.__code__.co_varnames:
        kwargs["max_categories"] = 50
    return OneHotEncoder(**kwargs)

################
### PIPELINE ###
################

# cleans df at begining
def _preclean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics and convert datetimes to seconds."""
    if not isinstance(df, pd.DataFrame):
        # Pass through untouched if this isn't a DataFrame
        return df
    out = df.copy()

    # datetimes -> seconds since epoch (float32)
    for c in out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        s = pd.to_datetime(out[c], errors="coerce")
        out[c] = (s.view("int64") / 1e9).astype("float32")

    # downcast numerics
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype("float32")
    for c in out.select_dtypes(include=["int64"]).columns:
        out[c] = out[c].astype("int32")
    return out

# used for PP pipeline
def _safety_dense_ct(X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=True, with_std=True)),
            ]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force dense array out
    )

# part of pipeline
def to_float32(A):
    return A.astype("float32")

#part of pipeline
class InfToNaN(BaseEstimator, TransformerMixin):
    """Replace ±inf with NaN in numeric arrays."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float, order="C", copy=True)
        X[~np.isfinite(X)] = np.nan
        return X

#minimal pipeline
def minimal_raw_pipeline(model: str, X: Any) -> Optional[Pipeline]:
    if not isinstance(X, pd.DataFrame):
        return None

    # Build a precleaner that preserves DataFrame (avoid 0-D array issues)
    preclean = FunctionTransformer(_preclean_df, validate=False, feature_names_out="one-to-one")

    # Use precleaned copy ONLY to *discover columns* (don’t call transform() inside steps)
    X_clean = _preclean_df(X)
    num_cols = X_clean.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = X_clean.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

    ohe = _make_ohe_sparse()

    if model.startswith("Random Forest"):
        transformers = []
        if num_cols:
            transformers.append((
                "num",
                Pipeline([
                    ("fix_inf", InfToNaN()),
                    ("imp", SimpleImputer(strategy="median")),
                    ("to32", FunctionTransformer(to_float32, validate=False)),
                ]),
                num_cols
            ))
        if cat_cols:
            transformers.append((
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", ohe),
                ]),
                cat_cols
            ))

        return Pipeline([
            ("preclean", preclean),  # stays as DataFrame thanks to validate=False
            ("ct", ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                sparse_threshold=0.3
            ))
        ])

    if model in ("Logistic Regression (LR)", "Support Vector Machine (SVM)", "Neural Network (MLP)"):
        transformers = []
        if num_cols:
            transformers.append((
                "num",
                Pipeline([
                    ("fix_inf", InfToNaN()),
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler(with_mean=True, with_std=True)),
                ]),
                num_cols
            ))
        if cat_cols:
            transformers.append((
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", ohe),
                ]),
                cat_cols
            ))

        return Pipeline([
            ("preclean", preclean),
            ("ct", ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                sparse_threshold=0.3
            ))
        ])

    # default: just preclean pass-through
    return Pipeline([("preclean", preclean)])


###############
### Metrics ###
###############

def fmt_bytes(n: int) -> str:
    if n is None or n < 0:
        return "n/a"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def summary_size_bytes(obj) -> int:
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return -1

def _predict_proba_safe(est, X):
    if hasattr(est, "predict_proba"):
        try:
            return est.predict_proba(X)
        except Exception:
            return None
    return None

def metrics_all(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "cls_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    # AUC if possible
    try:
        if y_prob is not None:
            if (y_prob.ndim == 1) or (y_prob.ndim == 2 and y_prob.shape[1] == 2):
                out["roc_auc"] = roc_auc_score(y_true, y_prob if y_prob.ndim == 1 else y_prob[:, 1])
            else:
                out["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        pass
    return out

def _pref(is_pipeline: bool, name: str) -> str:
    return f"clf__{name}" if is_pipeline else name

def fit_and_evaluate(
    estimator, *,
    X_train, y_train,
    X_val=None, y_val=None,
    X_test=None, y_test=None,
    sample_weight_train=None, sample_weight_val=None, sample_weight_test=None,
    gbm_engine: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
):
    # Detect task
    _ = type_of_target(y_train)  # raises if unsupported

    is_pipeline = isinstance(estimator, Pipeline)
    fit_params = {}

    # Sample weight to correct step
    if sample_weight_train is not None:
        fit_params[_pref(is_pipeline, "sample_weight")] = None if sample_weight_train is None else np.asarray(sample_weight_train).reshape(-1)

    # CatBoost cat_features
    if gbm_engine == "CatBoost" and _HAS_CAT:
        cat_idx = None
        if meta and isinstance(meta, dict):
            gbm_meta = (meta.get("gbm_categoricals") or {})
            cat_idx = gbm_meta.get("catboost_indices")
        if cat_idx is None and isinstance(X_train, pd.DataFrame):
            cat_idx = [i for i, dt in enumerate(X_train.dtypes) if str(dt) in ("object", "category")]
        if cat_idx:
            fit_params[_pref(is_pipeline, "cat_features")] = cat_idx

    # Train
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train, **fit_params)
    train_time = time.perf_counter() - t0

    # Evaluate helper
    def eval_split(X, y) -> Optional[Dict[str, Any]]:
        if X is None or y is None:
            return None
        t1 = time.perf_counter()
        y_pred = estimator.predict(X)
        infer_time = time.perf_counter() - t1
        y_prob = _predict_proba_safe(estimator, X)
        return {"metrics": metrics_all(y, y_pred, y_prob), "inference_time_sec": infer_time}

    results = {
        "model": estimator,
        "train": eval_split(X_train, y_train),
        "validation":   eval_split(X_val, y_val),
        "test":  eval_split(X_test, y_test),
        "train_time_sec": train_time,
        "model_size_bytes": summary_size_bytes(estimator),
        "n_features": getattr(X_train, "shape", [None, None])[1],
    }
    return results


#################
### ESTIMATOR ###
#################

def build_estimator(model: str, gbm_engine: Optional[str]):
    if model == "Random Forest (RF)":
        return RandomForestClassifier(
            n_estimators=100, 
            n_jobs=1, 
            random_state=42, 
            max_depth=18, 
            min_samples_leaf=2,
            max_features="sqrt",
            max_samples=0.7,
            bootstrap=True
        )

    if model == "Logistic Regression (LR)":
        return LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)

    if model == "Support Vector Machine (SVM)":
        return SVC(kernel="rbf", probability=True, random_state=42)

    if model == "Neural Network (MLP)":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)

    if model.startswith("Gradient Boosting"):
        if gbm_engine == "LightGBM" and _HAS_LGB:
            return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, random_state=42)
        if gbm_engine == "XGBoost" and _HAS_XGB:
            return xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, random_state=42, tree_method="hist")
        if gbm_engine == "CatBoost" and _HAS_CAT:
            return CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, verbose=False, random_state=42)

    # Fallback
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)


##############
### RUNNER ###
##############

def train_eval_orchestrator(
    *,
    model: str,
    use_preprocessed: bool,
    gbm_engine: str,
    # datasets (either preprocessed or raw)
    X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None,
    # optional weights (usually available for preprocessed flow)
    sample_weight_train=None, sample_weight_val=None, sample_weight_test=None,
    meta=None
) -> Dict[str, Any]:
    prep = None
    est = build_estimator(model, gbm_engine)

    if use_preprocessed and model == "Neural Network (MLP)":
        if isinstance(X_train, pd.DataFrame):
            # after dropping id-like columns
            has_obj = any(dtype == "object" or str(dtype) == "category" for dtype in X_train.dtypes)
            if has_obj:
                est = Pipeline([("safe", _safety_dense_ct(X_train)), ("clf", est)])
    
    if not use_preprocessed:
        prep = minimal_raw_pipeline(model, X_train)

    
    model = Pipeline([("prep", prep), ("clf", est)]) if prep is not None else est

    return fit_and_evaluate(
        model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        sample_weight_train=sample_weight_train,
        sample_weight_val=sample_weight_val,
        sample_weight_test=sample_weight_test,
        gbm_engine=gbm_engine,
        meta=meta,
    )


###############
### VISUALS ###
###############

# confusion matrix
def plot_confusion(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: Optional[Literal["true", "pred", "all"]] = "true",
    decimals: int = 2,
):
    cm = np.asarray(cm, dtype=np.float64)

    # Class labels
    if class_names is None:
        n = cm.shape[0]
        class_names = [str(i) for i in range(n)]

    # --- Normalization for color only ---
    if normalize == "true":          # row-wise (per actual)
        denom = cm.sum(axis=1, keepdims=True)
        cm_rel = np.divide(cm, np.maximum(denom, 1.0), where=denom != 0)
        cmax = 1.0
    elif normalize == "pred":        # column-wise (per predicted)
        denom = cm.sum(axis=0, keepdims=True)
        cm_rel = np.divide(cm, np.maximum(denom, 1.0), where=denom != 0)
        cmax = 1.0
    elif normalize == "all":         # global
        total = cm.sum()
        cm_rel = cm / total if total > 0 else cm.copy()
        cmax = 1.0
    else:                             # no normalization
        cm_rel = cm.copy()
        cmax = float(cm.max()) if cm.size else 1.0

    colorscale = [[0.0, "#FFFFFF"], [1.0, "#FF4B4B"]]

    # Show only counts in hover; z drives colors via normalized values
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_rel,
            x=class_names,
            y=class_names,
            coloraxis="coloraxis",
            customdata=cm.astype(int),
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{customdata}<extra></extra>",
        )
    )

    # Annotations: counts only (no percentages)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_rel[i, j]
            count_txt = f"{int(cm[i, j])}"
            # Improve contrast on dark cells
            font_color = "white" if val >= 0.6 * cmax else "black"
            fig.add_annotation(
                x=class_names[j],
                y=class_names[i],
                text=count_txt,
                showarrow=False,
                font=dict(color=font_color)
            )

    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        coloraxis=dict(colorscale=colorscale, cmin=0, cmax=cmax),
        height=420,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ROC
def plot_roc(model, X, y, class_names=None):
    from sklearn.metrics import roc_curve, auc

    try:
        probas = model.predict_proba(X)
    except Exception:
        return None

    classes = getattr(model, "classes_", None)
    if probas is None or classes is None:
        return None

    # If user provided class names, override
    if class_names is not None:
        if len(class_names) != len(classes):
            raise ValueError("Length of class_names must match number of classes in model")
    else:
        class_names = classes

    fig = go.Figure()

    # Binary case (special handling)
    if probas.ndim == 1 or (probas.ndim == 2 and probas.shape[1] == 2):
        p1 = probas if probas.ndim == 1 else probas[:, 1]
        fpr, tpr, _ = roc_curve(y, p1)
        roc_auc = auc(fpr, tpr)
        label = f"AUC={roc_auc:.3f}"
        # if we know names, display positive class name
        if len(class_names) == 2:
            label = f"{class_names[1]} (AUC={roc_auc:.3f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label))
    else:
        # Multi-class
        for k, cls in enumerate(classes):
            y_bin = (y == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, probas[:, k])
            roc_auc = auc(fpr, tpr)
            label = f"{class_names[k]} (AUC={roc_auc:.3f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label))

    # Chance line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))

    # Layout
    fig.update_layout(
        xaxis_title="FPR",
        yaxis_title="TPR",
        height=420,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

# PR
def plot_pr(model, X, y, class_names=None):

    try:
        probas = model.predict_proba(X)
    except Exception:
        return None

    classes = getattr(model, "classes_", None)
    if probas is None or classes is None:
        return None

    # If class_names not provided, just use classes from the model
    if class_names is None:
        class_names = [str(cls) for cls in classes]

    fig = go.Figure()

    # Binary / two-class case
    if probas.ndim == 1 or (probas.ndim == 2 and probas.shape[1] == 2):
        p1 = probas if probas.ndim == 1 else probas[:, 1]
        precision, recall, _ = precision_recall_curve(y, p1)
        ap = average_precision_score(y, p1)
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode="lines",
            name=f"{class_names[1] if len(class_names) > 1 else 'class 1'} (AP={ap:.3f})"
        ))
    else:
        # Multi-class
        for k, cls in enumerate(classes):
            y_bin = (y == cls).astype(int)
            precision, recall, _ = precision_recall_curve(y_bin, probas[:, k])
            ap = average_precision_score(y_bin, probas[:, k])
            fig.add_trace(go.Scatter(
                x=recall, y=precision, mode="lines",
                name=f"{class_names[k]} (AP={ap:.3f})"
            ))

    # Layout (no title)
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

# feature importance
def feature_importance_fig(model, feature_names=None, top_k=25):
    # Works if estimator or final step exposes feature_importances_
    clf = model
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]

    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return None

    idx = np.argsort(importances)[::-1][:top_k]
    imp_sorted = importances[idx]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    names_sorted = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]

    # Horizontal bar chart with custom color
    fig = go.Figure(
        go.Bar(
            x=imp_sorted[::-1],
            y=names_sorted[::-1],
            orientation="h",
            marker=dict(color="#FF4B4B")
        )
    )

    # Layout (no title)
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)  # reduced top margin
    )

    return fig

def _pp_delta(pre, raw):   # (pre - raw) in percentage points (0..1 -> 0..100 pp)
    if pre is None or raw is None:
        return None
    return (pre - raw) * 100.0

def _pct_delta(pre, raw):  # (pre - raw) / raw in %
    if pre in (None,) or raw in (None,) or raw == 0:
        return None
    return (pre - raw) / raw * 100.0

def render_compare_general(results_NP, results_P):
    st.subheader("General")

    tt_raw = results_NP.get("train_time_sec")
    tt_pre = results_P.get("train_time_sec")
    ms_raw = results_NP.get("model_size_bytes")
    ms_pre = results_P.get("model_size_bytes")
    nf_raw = results_NP.get("n_features")
    nf_pre = results_P.get("n_features")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Train time (s)**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RAW", f"{tt_raw:.3f}" if tt_raw is not None else "n/a")
        with c2:
            delta = _pct_delta(tt_pre, tt_raw) if (tt_pre is not None and tt_raw is not None) else None
            st.metric(
                "PRE",
                f"{tt_pre:.3f}" if tt_pre is not None else "n/a",
                f"{delta:+.2f}%" if delta is not None else "n/a",
                delta_color="inverse"  # lower is better
            )
    with col2:
        st.markdown("**Model size**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RAW", fmt_bytes(ms_raw) if ms_raw is not None else "n/a")
        with c2:
            delta = _pct_delta(ms_pre, ms_raw) if (ms_pre is not None and ms_raw is not None) else None
            st.metric(
                "PRE",
                fmt_bytes(ms_pre) if ms_pre is not None else "n/a",
                f"{delta:+.2f}%" if delta is not None else "n/a",
                delta_color="inverse"  # lower is better
            )
    with col3:
        st.markdown("**n-Features**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("RAW", f"{nf_raw}" if isinstance(nf_raw, (int, float)) else "n/a")
        with c2:
            if isinstance(nf_pre, (int, float)) and isinstance(nf_raw, (int, float)) and nf_pre not in (0, None):
                delta = (nf_pre - nf_raw) / nf_raw * 100
                st.metric("PRE", f"{nf_pre}", f"{delta:+.2f}%", delta_color="off")
            else:
                st.metric("PRE", f"{nf_pre}" if isinstance(nf_pre, (int, float)) else "n/a", "n/a", delta_color="off")

def render_compare_metrics(results_P, results_NP):
    st.subheader("Metrics")

    def _rows(results, tag):
        out = []
        for split in ["train", "validation", "test"]:
            r = results.get(split)
            if not r: 
                continue
            m = r["metrics"]
            out.append({
                "Split": split,
                "Tag": tag,
                "Accuracy": m.get("accuracy"),
                "F1 (macro)": m.get("f1_macro"),
                "Precision (macro)": m.get("precision_macro"),
                "Recall (macro)": m.get("recall_macro"),
                "ROC-AUC": m.get("roc_auc"),
                "Inference time [s]": r.get("inference_time_sec"),
            })
        return out

    rows = _rows(results_P, "RAW") + _rows(results_NP, "PRE")
    if not rows:
        st.info("No metrics available.")
        return

    df = pd.DataFrame(rows).set_index(["Split", "Tag"]).sort_index()

    rate_cols = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)", "ROC-AUC"]
    out_rows = []
    for split in ["train", "validation", "test"]:
        if split not in df.index.get_level_values(0):
            continue
        raw = df.loc[(split, "RAW")] if ("RAW" in df.loc[split].index) else None
        pre = df.loc[(split, "PRE")] if ("PRE" in df.loc[split].index) else None
        if raw is None or pre is None:
            continue

        row = {"Split": split}
        for c in rate_cols:
            row[(c, "RAW")] = raw[c]
            row[(c, "PRE")] = pre[c]
            row[(c, "Δ (pp)")] = _pp_delta(pre[c], raw[c])

        row[("Inference time [s]", "RAW")] = raw["Inference time [s]"]
        row[("Inference time [s]", "PRE")] = pre["Inference time [s]"]
        row[("Inference time [s]", "Δ %")] = _pct_delta(pre["Inference time [s]"], raw["Inference time [s]"])

        out_rows.append(row)

    if not out_rows:
        st.info("Need both RAW and PRE for at least one split.")
        return

    wide = pd.DataFrame(out_rows).set_index("Split")
    wide.columns = pd.MultiIndex.from_tuples(wide.columns)

    def fmt_rate(x):  return f"{x:.2%}" if pd.notnull(x) else "n/a"
    def fmt_pps(x):   return f"{x:+.2f} pp" if pd.notnull(x) else "n/a"
    def fmt_secs(x):  return f"{x:.4f}" if pd.notnull(x) else "n/a"
    def fmt_pct_(x):  return f"{x:+.1f}%" if pd.notnull(x) else "n/a"

    style_map = {}
    for c in rate_cols:
        style_map[(c, "RAW")] = fmt_rate
        style_map[(c, "PRE")] = fmt_rate
        style_map[(c, "Δ (pp)")] = fmt_pps
    style_map[("Inference time [s]", "RAW")] = fmt_secs
    style_map[("Inference time [s]", "PRE")] = fmt_secs
    style_map[("Inference time [s]", "Δ %")] = fmt_pct_

    def color_delta_rate(val):
        if pd.isnull(val): return ""
        if val > 0:   return "color: green; font-weight: bold;"
        if val < 0:   return "color: red; font-weight: bold;"
        return ""

    def color_delta_time(val):
        if pd.isnull(val): return ""
        if val < 0:   return "color: green; font-weight: bold;"
        if val > 0:   return "color: red; font-weight: bold;"
        return ""

    styled = (wide.style
        .format(style_map)
        .applymap(color_delta_rate, subset=[(c, "Δ (pp)") for c in rate_cols])
        .applymap(color_delta_time, subset=[("Inference time [s]", "Δ %")])
    )

    st.dataframe(styled, width="stretch")
