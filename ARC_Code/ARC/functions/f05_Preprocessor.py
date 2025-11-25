import streamlit as st

import functions.f00_Logger as logger

import numpy as np
import pandas as pd

from collections import Counter

# sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, Normalizer, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

# imbalanced
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


#####################
### MODEL PRESET ###
#####################

def modelPreset(choice, has_timestamp=False):
    if choice == "Random Forest (RF)":
        # Cleaning
        _DD         = True
        _MV         = True
        _MV_N       = "Median"
        _MV_C       = "Most Frequent"

        # Encoding (generic)
        _BR         = True
        _BR_MF      = 20
        _EN_GA      = True
        _EN_GA_O    = "One Hot Encoding"

        # GBM-native categorical prep (not used for RF)
        _GBM        = False
        _GBM_M      = "Light GBM"

        # Advanced encodings
        _CE         = False
        _TME        = False
        _TME_NS     = 5
        _TME_NOS    = 0.0
        _TME_GS     = 10.0

        # Feature filtering / selection
        _DC         = True
        _DC_V       = 0.0
        _DHC        = True
        _DHC_T      = 0.98
        _FS_RF      = True
        _FS_RF_TN   = 100
        _FS_MI      = False
        _FS_MI_TK   = 200
        _FS_A       = False
        _FS_A_TK    = 200
        _PCA        = False
        _PCA_N      = 0.95

        # Scaling & normalization
        _SC         = False
        _SC_S       = "Z-Score"
        _SC_S_MIN   = 0.0
        _SC_S_MAX   = 1.0
        _L2N        = False

        # Outliers & transforms
        _OC         = True
        _OC_M       = "IQR"
        _OC_WW      = 3.0
        _OC_PL      = 0.5
        _OC_PH      = 99.5
        _LOG        = False

        # Class imbalance
        _CW         = True
        _CW_S       = "balanced"
        _RS         = False
        _RS_M       = "SMOTE"
        _RS_SK      = 5

        # SVM kernel approximations (N/A)
        _SVM        = False
        _SVM_M      = "Nyström"
        _SVM_M_NNC  = 2000
        _SVM_M_RFFNC= 2000

        # Time features
        _TF         = has_timestamp

    elif choice == "Gradient Boosting (XGBoost / LightGBM / CatBoost)":
        # Cleaning
        _DD         = True
        _MV         = True
        _MV_N       = "Median"
        _MV_C       = "Most Frequent"

        # Encoding: prefer GBM-native, not generic
        _BR         = True
        _BR_MF      = 20
        _EN_GA      = False
        _EN_GA_O    = "One Hot Encoding"
        _GBM        = True 
        _GBM_M      = "Light GBM"

        # Advanced encodings
        _CE         = False
        _TME        = False
        _TME_NS     = 5
        _TME_NOS    = 0.0
        _TME_GS     = 10.0

        # Feature filtering / selection
        _DC         = True
        _DC_V       = 0.0
        _DHC        = True
        _DHC_T      = 0.98
        _FS_RF      = False
        _FS_RF_TN   = 100
        _FS_MI      = False
        _FS_MI_TK   = 200
        _FS_A       = False
        _FS_A_TK    = 200
        _PCA        = False
        _PCA_N      = 0.95

        # Scaling & normalization
        _SC         = False
        _SC_S       = "Z-Score"
        _SC_S_MIN   = 0.0
        _SC_S_MAX   = 1.0
        _L2N        = False

        # Outliers & transforms
        _OC         = True
        _OC_M       = "IQR"
        _OC_WW      = 3.0
        _OC_PL      = 0.5
        _OC_PH      = 99.5
        _LOG        = False

        # Class imbalance
        _CW         = True
        _CW_S       = "balanced"
        _RS         = False
        _RS_M       = "SMOTE"
        _RS_SK      = 5

        # SVM kernel approximations (N/A)
        _SVM        = False
        _SVM_M      = "Nyström"
        _SVM_M_NNC  = 2000
        _SVM_M_RFFNC= 2000

        # Time features
        _TF         = has_timestamp

    elif choice == "Support Vector Machine (SVM)":
        # Cleaning
        _DD         = True
        _MV         = True
        _MV_N       = "Median"
        _MV_C       = "Most Frequent"

        # Encoding (generic)
        _BR         = True
        _BR_MF      = 20
        _EN_GA      = True
        _EN_GA_O    = "One Hot Encoding"
        _GBM        = False
        _GBM_M      = "Light GBM"

        # Advanced encodings
        _CE         = False
        _TME        = False
        _TME_NS     = 5
        _TME_NOS    = 0.0
        _TME_GS     = 10.0

        # Feature filtering / selection
        _DC         = True
        _DC_V       = 0.0
        _DHC        = True
        _DHC_T      = 0.98
        _FS_RF      = False
        _FS_RF_TN   = 100
        _FS_MI      = True
        _FS_MI_TK   = 200
        _FS_A       = True
        _FS_A_TK    = 200
        _PCA        = False
        _PCA_N      = 0.95

        # Scaling & normalization
        _SC         = True
        _SC_S       = "Z-Score"
        _SC_S_MIN   = 0.0
        _SC_S_MAX   = 1.0
        _L2N        = False

        # Outliers & transforms
        _OC         = True
        _OC_M       = "IQR"
        _OC_WW      = 3.0
        _OC_PL      = 0.5
        _OC_PH      = 99.5
        _LOG        = False

        # Class imbalance
        _CW         = True
        _CW_S       = "balanced"
        _RS         = False
        _RS_M       = "SMOTE"
        _RS_SK      = 5

        # SVM kernel approximations (off by default; turn on for huge data)
        _SVM        = False
        _SVM_M      = "Nyström"
        _SVM_M_NNC  = 2000
        _SVM_M_RFFNC= 2000

        # Time features
        _TF         = has_timestamp

    elif choice == "Neural Network (MLP)":
        # Cleaning
        _DD         = True
        _MV         = True
        _MV_N       = "Median"
        _MV_C       = "Most Frequent"

        # Encoding (generic)
        _BR         = True
        _BR_MF      = 20
        _EN_GA      = True
        _EN_GA_O    = "One Hot Encoding"
        _GBM        = False
        _GBM_M      = "Light GBM"

        # Advanced encodings
        _CE         = False
        _TME        = False
        _TME_NS     = 5
        _TME_NOS    = 0.0
        _TME_GS     = 10.0

        # Feature filtering / selection
        _DC         = True
        _DC_V       = 0.0
        _DHC        = False
        _DHC_T      = 0.98
        _FS_RF      = False
        _FS_RF_TN   = 100
        _FS_MI      = False
        _FS_MI_TK   = 200
        _FS_A       = False
        _FS_A_TK    = 200
        _PCA        = False
        _PCA_N      = 0.95

        # Scaling & normalization
        _SC         = True
        _SC_S       = "Z-Score"
        _SC_S_MIN   = 0.0
        _SC_S_MAX   = 1.0
        _L2N        = False

        # Outliers & transforms
        _OC         = True
        _OC_M       = "IQR"
        _OC_WW      = 3.0
        _OC_PL      = 0.5
        _OC_PH      = 99.5
        _LOG        = True

        # Class imbalance
        _CW         = True
        _CW_S       = "balanced"
        _RS         = False
        _RS_M       = "SMOTE"
        _RS_SK      = 5

        # SVM kernel approximations (N/A)
        _SVM        = False
        _SVM_M      = "Nyström"
        _SVM_M_NNC  = 2000
        _SVM_M_RFFNC= 2000

        # Time features
        _TF         = has_timestamp

    elif choice == "Logistic Regression (LR)":
        # Cleaning
        _DD         = True
        _MV         = True
        _MV_N       = "Median"
        _MV_C       = "Most Frequent"

        # Encoding (generic)
        _BR         = True
        _BR_MF      = 20
        _EN_GA      = True
        _EN_GA_O    = "One Hot Encoding"
        _GBM        = False
        _GBM_M      = "Light GBM"

        # Advanced encodings
        _CE         = False
        _TME        = False
        _TME_NS     = 5
        _TME_NOS    = 0.0
        _TME_GS     = 10.0

        # Feature filtering / selection
        _DC         = True
        _DC_V       = 0.0
        _DHC        = True
        _DHC_T      = 0.98
        _FS_RF      = False
        _FS_RF_TN   = 100
        _FS_MI      = True
        _FS_MI_TK   = 200
        _FS_A       = True
        _FS_A_TK    = 200
        _PCA        = False
        _PCA_N      = 0.95

        # Scaling & normalization
        _SC         = True
        _SC_S       = "Z-Score"
        _SC_S_MIN   = 0.0
        _SC_S_MAX   = 1.0
        _L2N        = False

        # Outliers & transforms
        _OC         = True
        _OC_M       = "IQR"
        _OC_WW      = 3.0
        _OC_PL      = 0.5
        _OC_PH      = 99.5
        _LOG        = True

        # Class imbalance
        _CW         = True
        _CW_S       = "balanced"
        _RS         = False
        _RS_M       = "SMOTE"
        _RS_SK      = 5

        # SVM kernel approximations (N/A)
        _SVM        = False
        _SVM_M      = "Nyström"
        _SVM_M_NNC  = 2000
        _SVM_M_RFFNC= 2000

        # Time features
        _TF         = has_timestamp

    elif choice == "Local Outlier Factor (LOF)":
        # Cleaning
        _DD         = True
        _MV         = True
        _MV_N       = "Median"
        _MV_C       = "Most Frequent"

        # Encoding (keep distances meaningful; control dimensionality)
        _BR         = True
        _BR_MF      = 15
        _EN_GA      = True
        _EN_GA_O    = "One Hot Encoding"
        _GBM        = False
        _GBM_M      = "Light GBM"

        # Advanced encodings (disable label-driven/biased ones)
        _CE         = False
        _TME        = False
        _TME_NS     = 5
        _TME_NOS    = 0.0
        _TME_GS     = 10.0

        # Feature filtering / redundancy control
        _DC         = True
        _DC_V       = 0.0
        _DHC        = True
        _DHC_T      = 0.95
        _FS_RF      = False
        _FS_RF_TN   = 100
        _FS_MI      = False
        _FS_MI_TK   = 200
        _FS_A       = False
        _FS_A_TK    = 200

        # Dimensionality reduction (helps LOF in high-D)
        _PCA        = True
        _PCA_N      = 0.90

        # Scaling & normalization (critical for distance-based methods)
        _SC         = True
        _SC_S       = "Robust" 
        _SC_S_MIN   = 0.0
        _SC_S_MAX   = 1.0
        _L2N        = False 

        # Outliers & transforms
        _OC         = False 
        _OC_M       = "IQR"
        _OC_WW      = 3.0
        _OC_PL      = 0.5
        _OC_PH      = 99.5
        _LOG        = False 

        # Class imbalance (not applicable in unsupervised LOF)
        _CW         = False
        _CW_S       = "balanced"
        _RS         = False
        _RS_M       = "SMOTE"
        _RS_SK      = 5

        # Kernel approximations (irrelevant)
        _SVM        = False
        _SVM_M      = "Nyström"
        _SVM_M_NNC  = 2000
        _SVM_M_RFFNC= 2000

        # Time features (keep whatever you already detect)
        _TF         = has_timestamp
    else:
        # Cleaning
        _DD         = False
        _MV         = False
        _MV_N       = "Median"
        _MV_C       = "Most Frequent"

        # Encoding (generic)
        _BR         = False
        _BR_MF      = 20
        _EN_GA      = False
        _EN_GA_O    = "One Hot Encoding"

        # GBM-native categorical prep (not used for RF)
        _GBM        = False
        _GBM_M      = "Light GBM"

        # Advanced encodings
        _CE         = False
        _TME        = False
        _TME_NS     = 5
        _TME_NOS    = 0.0
        _TME_GS     = 10.0

        # Feature filtering / selection
        _DC         = False
        _DC_V       = 0.0
        _DHC        = False
        _DHC_T      = 0.98
        _FS_RF      = False
        _FS_RF_TN   = 100
        _FS_MI      = False
        _FS_MI_TK   = 200
        _FS_A       = False
        _FS_A_TK    = 200
        _PCA        = False
        _PCA_N      = 0.95

        # Scaling & normalization
        _SC         = False
        _SC_S       = "Z-Score"
        _SC_S_MIN   = 0.0
        _SC_S_MAX   = 1.0
        _L2N        = False

        # Outliers & transforms
        _OC         = False
        _OC_M       = "IQR"
        _OC_WW      = 3.0
        _OC_PL      = 0.5
        _OC_PH      = 99.5
        _LOG        = False

        # Class imbalance
        _CW         = False
        _CW_S       = "balanced"
        _RS         = False
        _RS_M       = "SMOTE"
        _RS_SK      = 5

        # SVM kernel approximations (N/A)
        _SVM        = False
        _SVM_M      = "Nyström"
        _SVM_M_NNC  = 2000
        _SVM_M_RFFNC= 2000

        # Time features
        _TF         = False

    return{
        "DD": _DD,
        "MV": _MV,
        "MV_N": _MV_N,
        "MV_C": _MV_C,
        "BR": _BR,
        "BR_MF": _BR_MF,
        "EN_GA": _EN_GA,
        "EN_GA_O": _EN_GA_O,
        "GBM": _GBM,
        "GBM_M": _GBM_M,
        "CE": _CE,
        "TME": _TME,
        "TME_NS": _TME_NS,
        "TME_NOS": _TME_NOS,
        "TME_GS": _TME_GS,
        "DC": _DC,
        "DC_V": _DC_V,
        "DHC": _DHC,
        "DHC_T": _DHC_T,
        "FS_RF": _FS_RF,
        "FS_RF_TN": _FS_RF_TN,
        "FS_MI": _FS_MI,
        "FS_MI_TK": _FS_MI_TK,
        "FS_A": _FS_A,
        "FS_A_TK": _FS_A_TK,
        "PCA": _PCA,
        "PCA_N": _PCA_N,
        "SC": _SC,
        "SC_S": _SC_S,
        "SC_S_MIN": _SC_S_MIN,
        "SC_S_MAX": _SC_S_MAX,
        "L2N": _L2N,
        "OC": _OC,
        "OC_M": _OC_M,
        "OC_WW": _OC_WW,
        "OC_PL": _OC_PL,
        "OC_PH": _OC_PH,
        "LOG": _LOG,
        "CW": _CW,
        "CW_S": _CW_S,
        "RS": _RS,
        "RS_M": _RS_M,
        "RS_SK": _RS_SK,
        "SVM": _SVM,
        "SVM_M": _SVM_M,
        "SVM_M_NNC": _SVM_M_NNC,
        "SVM_M_RFFNC": _SVM_M_RFFNC,
        "TF": _TF
    }
 

###############################
### PREPROCESSING FUNCTIONS ###
###############################

### label encoding ################
def encode_labels(y_train, y_val=None, y_test=None):
    # Encode categorical labels to integers based on TRAIN only.
    ytr = pd.Series(y_train).copy()
    classes = pd.Index(sorted(pd.Series(ytr).unique()))
    mapping = {c: i for i, c in enumerate(classes)}
    ytr_enc = ytr.map(mapping)
    yva_enc = None if y_val is None else pd.Series(y_val).map(mapping)
    yte_enc = None if y_test is None else pd.Series(y_test).map(mapping)
    
    return ytr_enc, yva_enc, yte_enc, mapping


### cleaning ##################

# drop duplicates
def drop_duplicates(
        X_train, y_train=None, 
        X_val=None, y_val=None, 
        X_test=None, y_test=None
):
    # drop duplicate feature rows

    # train
    idx_tr = ~X_train.duplicated(keep="first")
    Xtr = X_train.loc[idx_tr].copy()
    ytr = y_train.loc[idx_tr].copy() if y_train is not None else None
    dropped_tr = int((~idx_tr).sum())

    # val
    if X_val is not None:
        idx_va = ~X_val.duplicated(keep="first")
        Xva = X_val.loc[idx_va].copy()
        yva = y_val.loc[idx_va].copy() if y_val is not None else None
        dropped_va = int((~idx_va).sum())
    else:
        Xva, yva, dropped_va = None, None, 0

    # test
    if X_test is not None:
        idx_te = ~X_test.duplicated(keep="first")
        Xte = X_test.loc[idx_te].copy()
        yte = y_test.loc[idx_te].copy() if y_test is not None else None
        dropped_te = int((~idx_te).sum())
    else:
        Xte, yte, dropped_te = None, None, 0

    # stats
    stats = {
        "train_total": len(X_train), 
        "train_kept": int(idx_tr.sum()), 
        "train_dropped": dropped_tr,
        "val_total": 0 if X_val is None else len(X_val),
        "val_kept": 0 if X_val is None else int(idx_va.sum()),
        "val_dropped": dropped_va,
        "test_total": 0 if X_test is None else len(X_test),
        "test_kept": 0 if X_test is None else int(idx_te.sum()),
        "test_dropped": dropped_te,
    }

    # log
    if logger is not None and hasattr(logger, "save_log"):
        logger.save_log(f"Preprocessing - drop duplicates: train({stats['train_dropped']}), val ({stats['val_dropped']}), test({stats['test_dropped']})")

    return Xtr, ytr, Xva, yva, Xte, yte, stats

# impute missing values
def impute_numeric_categorical(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    num_strategy: str = "median",
    cat_strategy: str = "most_frequent"
):
    
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # detect numeric and categorical features
    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat = Xtr.select_dtypes(exclude=[np.number]).columns.tolist()

    # sanitize numeric columns: inf -> NaN
    if num:
        if Xtr is not None:
            Xtr = Xtr.copy()
            Xtr[num] = Xtr[num].replace([np.inf, -np.inf], np.nan)
        if Xva is not None:
            Xva = Xva.copy()
            Xva[num] = Xva[num].replace([np.inf, -np.inf], np.nan)
        if Xte is not None:
            Xte = Xte.copy()
            Xte[num] = Xte[num].replace([np.inf, -np.inf], np.nan)

    # drop all-NaN numeric columns
    droped_cols = []
    if num:
        all_nan_cols = Xtr[num].isna().all()
        drop_num = all_nan_cols[all_nan_cols].index.tolist()
        if drop_num:
            droped_cols.append(drop_num)
            Xtr.drop(columns=drop_num, inplace=True)
            if Xva is not None: Xva.drop(columns=drop_num, inplace=True, errors="ignore")
            if Xte is not None: Xte.drop(columns=drop_num, inplace=True, errors="ignore")
            num = [c for c in num if c not in drop_num]

    # fit imputers
    impn = SimpleImputer(strategy=num_strategy) if num else None
    impc = SimpleImputer(strategy=cat_strategy) if cat else None

    # numeric impute
    if num:
        Xtr[num] = impn.fit_transform(Xtr[num])
        if Xva is not None: Xva[num] = impn.transform(Xva[num])
        if Xte is not None: Xte[num] = impn.transform(Xte[num])

    # categorical impute
    if cat:
        Xtr[cat] = impc.fit_transform(Xtr[cat])
        if Xva is not None: Xva[cat] = impc.transform(Xva[cat])
        if Xte is not None: Xte[cat] = impc.transform(Xte[cat])

    #log
    logger.save_log(f"Preprocessing - impute missing values: Missing values imputated with {num_strategy} (numeric) and {cat_strategy} (categorical), dropped {droped_cols} column(s) due to all-NaN")
    return Xtr, Xva, Xte, impn, impc


### encoding ##################

# bucket rare categories
def bucket_rare_categories(X_train, X_val=None, X_test=None, min_freq=20, other_token='__OTHER__'):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()
    # identify categorical columns
    cat = Xtr.select_dtypes(exclude=[np.number]).columns.tolist()
    # process each categorical column
    maps = {}
    for c in cat:
        # get value counts
        vc = Xtr[c].value_counts(dropna=False)
        # determine categories to keep
        keep = set(vc[vc >= min_freq].index)
        # store mapping
        maps[c] = keep

        # replace infrequent categories with other_token
        Xtr[c] = Xtr[c].where(Xtr[c].isin(keep), other_token)
        if Xva is not None: Xva[c] = Xva[c].where(Xva[c].isin(keep), other_token)
        if Xte is not None: Xte[c] = Xte[c].where(Xte[c].isin(keep), other_token)

    logger.save_log(f"Preprocessing - bucket rare categories: rere categories are bucketed (min_freq={min_freq})")

    return Xtr, Xva, Xte, maps

# one-hot encoding
def encode_one_hot(X_train, X_val=None, X_test=None, handle_unknown='ignore'):
    Xtr = X_train.copy()

    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat = Xtr.select_dtypes(exclude=[np.number]).columns.tolist()

    # if no categoricals, return copies
    if not cat:
        return (
            Xtr,
            None if X_val is None else X_val.copy(),
            None if X_test is None else X_test.copy(),
            None,
        )

    # fit OHE on train
    ohe = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    tr_cat = ohe.fit_transform(Xtr[cat])
    cat_cols = ohe.get_feature_names_out(cat)

    # construct encoded train set – keep original index for BOTH parts
    Xtr_enc = pd.concat(
        [
            Xtr[num],  # no reset_index here
            pd.DataFrame(tr_cat, columns=cat_cols, index=Xtr.index),
        ],
        axis=1,
    )

    # transform val
    if X_val is None:
        Xva_enc = None
    else:
        arr_va = ohe.transform(X_val[cat])
        Xva_enc = pd.concat(
            [
                X_val[num],  # no reset_index
                pd.DataFrame(arr_va, columns=cat_cols, index=X_val.index),
            ],
            axis=1,
        )

    # transform test
    if X_test is None:
        Xte_enc = None
    else:
        arr_te = ohe.transform(X_test[cat])
        Xte_enc = pd.concat(
            [
                X_test[num],  # no reset_index
                pd.DataFrame(arr_te, columns=cat_cols, index=X_test.index),
            ],
            axis=1,
        )

    # align val/test to train columns
    _ref_cols = Xtr_enc.columns
    if Xva_enc is not None:
        Xva_enc = Xva_enc.reindex(columns=_ref_cols, fill_value=0)
    if Xte_enc is not None:
        Xte_enc = Xte_enc.reindex(columns=_ref_cols, fill_value=0)

    logger.save_log("Preprocessing - one-hot encoding: One-hot encoded")
    return Xtr_enc, Xva_enc, Xte_enc, ohe

# ordinal encoding
def encode_ordinal(X_train, X_val=None, X_test=None, categories='auto'):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()
    # identify categorical columns
    cat = Xtr.select_dtypes(exclude=[np.number]).columns.tolist()
    if not cat:
        return Xtr, Xva, Xte, None
    # fit ordinal encoder on train
    ord_enc = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
    # encode train
    Xtr[cat] = ord_enc.fit_transform(Xtr[cat])
    # encode val/test
    if Xva is not None: Xva[cat] = ord_enc.transform(Xva[cat])
    if Xte is not None: Xte[cat] = ord_enc.transform(Xte[cat])
    # log
    logger.save_log(f"Preprocessing - ordinal encoding: Categoricals ordinal encoded")

    return Xtr, Xva, Xte, ord_enc

# gbm-native categorical prep
def prepare_categoricals_gbm(X_train, X_val=None, X_test=None, backend="catboost"):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()
    # identify categorical columns
    cat_cols = Xtr.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    if backend.lower() == "catboost":
        # CatBoost: convert to string
        for c in cat_cols:
            Xtr[c] = Xtr[c].astype(str)
            if Xva is not None: Xva[c] = Xva[c].astype(str)
            if Xte is not None: Xte[c] = Xte[c].astype(str)
        cat_idx = [Xtr.columns.get_loc(c) for c in cat_cols]
        
        logger.save_log(f"Preprocessing - prepare for GBM: prepared categorical columns for CatBoost")
        return Xtr, Xva, Xte, cat_idx

    for c in cat_cols:
        # lightGBM/XGBoost: convert to categorical with fixed categories from train
        cats = pd.Categorical(Xtr[c]).categories
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats)
        if Xva is not None: Xva[c] = pd.Categorical(Xva[c], categories=cats)
        if Xte is not None: Xte[c] = pd.Categorical(Xte[c], categories=cats)

    logger.save_log(f"Preprocessing - prepare for GBM: prepared categorical columns for LightGBM/XGBoost")
    return Xtr, Xva, Xte, cat_cols

# count encoding
def count_encoding(OG_X_train, X_train, X_val=None, X_test=None, default=0.0, suffix='_cnt'):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # pick categorical columns from current frame; else fall back to original train
    cur_cat = Xtr.select_dtypes(exclude=[np.number]).columns.tolist()
    if cur_cat:
        cols = cur_cat
    else:
        _, og_cat = OG_X_train.select_dtypes(exclude=[np.number]).columns.tolist()
        cols = og_cat

    maps = {}
    if cols:
        # fit on train
        for c in cols:
            if c in Xtr.columns:
                vc = Xtr[c].value_counts(dropna=False)
                maps[c] = vc.to_dict()
        # train
        for c, m in maps.items():
            if c in Xtr.columns:
                Xtr[c + suffix] = Xtr[c].map(m).fillna(default).astype(float)
        # val
        if Xva is not None:
            for c, m in maps.items():
                if c in Xva.columns:
                    Xva[c + suffix] = Xva[c].map(m).fillna(default).astype(float)
        # test
        if Xte is not None:
            for c, m in maps.items():
                if c in Xte.columns:
                    Xte[c + suffix] = Xte[c].map(m).fillna(default).astype(float)

    logger.save_log(f"Preprocessing - count encoding: Count encoded categorical columns {list(maps.keys())}")
    return Xtr, Xva, Xte, cols, maps

# target mean encoding
def target_mean_encoding_kfold(
        X_train, y_train, X_val=None, X_test=None,
        cols=None, n_splits=5, noise_std=0.0, global_smoothing=10.0, random_state=42):
    if cols is None:
        cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()
    # setup random state
    rng = np.random.RandomState(random_state)
    # precompute global mean
    global_mean = y_train.mean()
    # k-fold split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # encode
    enc_maps = {}
    for c in cols:
        # out-of-fold encoding for train
        oof = pd.Series(index=Xtr.index, dtype=float)
        # fold-wise
        for tr_idx, val_idx in kf.split(Xtr):
            # compute target mean mapping on train fold
            tr_cats = Xtr.iloc[tr_idx][c]
            tr_y = y_train.iloc[tr_idx]
            # compute smoothed means
            counts = tr_cats.value_counts()
            means = tr_y.groupby(tr_cats).mean()
            smooth = (means * counts + global_mean * global_smoothing) / (counts + global_smoothing)
            # map to val fold
            oof.iloc[val_idx] = Xtr.iloc[val_idx][c].map(smooth).fillna(global_mean).values
        # add noise if specified
        if noise_std > 0:
            oof += rng.normal(0, noise_std, size=len(oof))
        # assign to train
        Xtr[c + '_tgtmean'] = oof
        # fit full mapping on entire train for val/test
        counts_full = Xtr[c].value_counts()
        means_full = y_train.groupby(Xtr[c]).mean()
        smooth_full = (means_full * counts_full + global_mean * global_smoothing) / (counts_full + global_smoothing)
        enc_maps[c] = (smooth_full.to_dict(), global_mean)
        # transform val/test
        if Xva is not None:
            m, g = enc_maps[c]
            Xva[c + '_tgtmean'] = Xva[c].map(m).fillna(g).astype(float)
        if Xte is not None:
            m, g = enc_maps[c]
            Xte[c + '_tgtmean'] = Xte[c].map(m).fillna(g).astype(float)
    logger.save_log(f"Preprocessing - target mean encoding: Target mean encoded columns {cols} with {n_splits}-fold out-of-fold strategy")
    return Xtr, Xva, Xte, enc_maps


### feature pruning ###########

# drop constant/low variance
def drop_low_variance(X_train, X_val=None, X_test=None, threshold=0.0, drop_constant_non_numeric=True):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # separate numeric and non-numeric columns
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    non_num_cols = [c for c in Xtr.columns if c not in num_cols]

    kept_num, removed_num = [], []
    if num_cols:
        # convert to numeric, coerce errors
        Xnum = Xtr[num_cols].apply(pd.to_numeric, errors="coerce")
        Xnum = Xnum.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # apply VarianceThreshold
        vt = VarianceThreshold(threshold=threshold)
        Xnum_arr = vt.fit_transform(Xnum)
        kept_mask = vt.get_support()
        # get kept/removed numeric columns
        kept_num = [col for col, keep in zip(num_cols, kept_mask) if keep]
        removed_num = [col for col, keep in zip(num_cols, kept_mask) if not keep]

    # process non-numeric columns
    kept_non_num, removed_non_num = [], []
    if drop_constant_non_numeric and non_num_cols:
        for c in non_num_cols:
            # check uniqueness
            if Xtr[c].nunique(dropna=False) > 1:
                kept_non_num.append(c)
            else:
                removed_non_num.append(c)
    else:
        kept_non_num = non_num_cols

    # combine kept/removed columns
    kept_set = set(kept_num) | set(kept_non_num)
    kept = [c for c in Xtr.columns if c in kept_set]

    removed = removed_num + removed_non_num

    # reindex all splits
    Xtr2 = None if Xtr is None else Xtr.reindex(columns=kept, fill_value=0)
    Xva2 = None if Xva is None else Xva.reindex(columns=kept, fill_value=0)
    Xte2 = None if Xte is None else Xte.reindex(columns=kept, fill_value=0)

    logger.save_log(f"Preprocessing - drop low variance: Dropped {len(removed)} low-variance columns: {removed}")

    return Xtr2, Xva2, Xte2, kept, removed

# drop highly correlated numeric features
def drop_high_corr_numeric(X_train, X_val=None, X_test=None, threshold=0.98):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()
    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    if not num:
        return Xtr, Xva, Xte, Xtr.columns.tolist(), []
    # compute correlation matrix
    corr = Xtr[num].corr().abs()
    # upper triangle mask
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # identify columns to drop
    to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
    # retain columns
    keep = [c for c in Xtr.columns if c not in to_drop]

    logger.save_log(f"Preprocessing - drop high correlation: Dropped {len(to_drop)} highly correlated numeric columns: {to_drop}")

    return Xtr[keep].copy(), None if Xva is None else Xva[keep].copy(), None if Xte is None else Xte[keep].copy(), keep, to_drop

# RF-specific feature selection
def select_features_by_rf_importance(
    X_train, y_train, X_val=None, X_test=None,
    n_keep=50, random_state=42, n_estimators=300, max_depth=None,
    datetime_policy: str = "unix",   # "drop" | "unix"
):
    # copy inputs
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # prepare data for RF fitting
    # datetime handling
    dt_cols = Xtr.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    if dt_cols:
        if datetime_policy == "unix":
            # convert to UNIX seconds (float)
            if Xtr is not None:
                Xtr = Xtr.copy()
                for c in dt_cols:
                    ts = pd.to_datetime(Xtr[c], errors="coerce", utc=True)
                    Xtr[c] = (ts.view("int64") / 1e9).astype("float64")
            if Xva is not None:
                Xva = Xva.copy()
                for c in dt_cols:
                    ts = pd.to_datetime(Xva[c], errors="coerce", utc=True)
                    Xva[c] = (ts.view("int64") / 1e9).astype("float64")
            if Xte is not None:
                Xte = Xte.copy()
                for c in dt_cols:
                    ts = pd.to_datetime(Xte[c], errors="coerce", utc=True)
                    Xte[c] = (ts.view("int64") / 1e9).astype("float64")
        else:
            # drop datetimes for RF fit
            Xtr = Xtr.drop(columns=dt_cols)
            if Xva is not None: Xva = Xva.drop(columns=dt_cols, errors="ignore")
            if Xte is not None: Xte = Xte.drop(columns=dt_cols, errors="ignore")

    # keep only numeric/bool columns
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    if not num_cols:
        # no numeric features: return empty selection
        importances = pd.Series(dtype=float)
        kept = []
        removed = X_train.columns.tolist()
        return (X_train.copy(),
                None if X_val is None else X_val.copy(),
                None if X_test is None else X_test.copy(),
                kept, importances, removed)

    # clean numeric data: convert to numeric, coerce errors, handle inf
    if Xtr is None: 
        Xnum_tr=None
    else:
        Xnum_tr = Xtr[num_cols].apply(pd.to_numeric, errors="coerce")
        Xnum_tr = Xnum_tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # fit RF to get feature importances
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight='balanced_subsample'
    )
    rf.fit(Xnum_tr, y_train)
    importances = pd.Series(rf.feature_importances_, index=num_cols)

    # select top n_keep features
    kept_numeric = importances.sort_values(ascending=False).head(n_keep).index.tolist()

    # build final kept/removed lists
    kept = [c for c in X_train.columns if c in kept_numeric]
    removed = [c for c in X_train.columns if c not in kept]

    # reindex all splits
    Xtr2 = None if Xtr is None else Xtr.reindex(columns=kept, fill_value=0)
    Xva2 = None if X_val is None else X_val.reindex(columns=kept, fill_value=0)
    Xte2 = None if X_test is None else X_test.reindex(columns=kept, fill_value=0)

    logger.save_log(f"Preprocessing - RF feature selection: Selected top {n_keep} features by Random Forest importance: {kept_numeric}")

    return Xtr2, Xva2, Xte2, kept, importances, removed

# feature selection by mutual information
def select_by_mutual_info(
    X_train, y_train, X_val=None, X_test=None,
    k=200, random_state=42,
    use_factorize_categoricals: bool = False,  # False = ignore non-numeric; True = factorize & treat as discrete
    
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # build candidate columns for mutual information
    cont_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    disc_cols = []

    if use_factorize_categoricals:
        cat_cols = Xtr.select_dtypes(include=["object", "category"]).columns.tolist()
        # factorize categoricals into int codes for mutual information (treated as discrete)
        for c in cat_cols:
            codes, _ = pd.factorize(Xtr[c], sort=True, na_sentinel=-1)
            Xtr[c] = codes.astype("int64")
            if Xva is not None:
                codes_v = pd.Categorical(Xva[c], categories=np.unique(Xtr[c])).codes
                Xva[c] = codes_v.astype("int64")
            if Xte is not None:
                codes_t = pd.Categorical(Xte[c], categories=np.unique(Xtr[c])).codes
                Xte[c] = codes_t.astype("int64")
        disc_cols = cat_cols

    # final candidate columns for MI
    cols_used = cont_cols + disc_cols
    if not cols_used:
        # nothing to score; return as-is
        mi_series = pd.Series(dtype=float)
        kept = []
        removed = X_train.columns.tolist()
        return X_train.copy(), (None if X_val is None else X_val.copy()), (None if X_test is None else X_test.copy()), kept, mi_series, removed
        
    # clean numeric block
    Xnum = Xtr[cols_used].apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # discrete mask for MI
    if use_factorize_categoricals:
        discrete_mask = np.array([c in disc_cols for c in cols_used], dtype=bool)
        mi = mutual_info_classif(Xnum.values, y_train, random_state=random_state, discrete_features=discrete_mask)
    else:
        mi = mutual_info_classif(Xnum.values, y_train, random_state=random_state)

    mi_series = pd.Series(mi, index=cols_used)

    # choose top-k from cols_used
    k_eff = min(k, len(cols_used))
    kept_used = mi_series.sort_values(ascending=False).head(k_eff).index.tolist()

    # project to kept (preserving original order among all columns)
    kept = [c for c in X_train.columns if c in kept_used]
    removed = [c for c in X_train.columns if c not in kept]

    # reindex all splits
    Xtr2 = None if Xtr is None else Xtr.reindex(columns=kept, fill_value=0)
    Xva2 = None if Xva is None else Xva.reindex(columns=kept, fill_value=0)
    Xte2 = None if Xte is None else Xte.reindex(columns=kept, fill_value=0)

    logger.save_log(f"Preprocessing - mutual information feature selection: Selected top {k_eff} features by mutual information: {kept_used}")

    return Xtr2, Xva2, Xte2, kept, mi_series, removed

# feature selection by ANOVA F-score
def select_by_anova_f(
    X_train, y_train, X_val=None, X_test=None, k=200,
    drop_zero_variance: bool = True, # drop numeric cols with variance == 0 before ANOVA
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # select numeric/bool columns
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    if not num_cols:
        # nothing to score; return as-is
        f_series = pd.Series(dtype=float, index=X_train.columns)
        kept = []
        removed = X_train.columns.tolist()
        return X_train.copy(), (None if X_val is None else X_val.copy()), (None if X_test is None else X_test.copy()), kept, f_series, removed

    # clean numeric data: convert to numeric, coerce errors, handle inf, impute NaN with median from train
    def _clean_numeric(df, fit=False, medians=None):
        if df is None: return None, medians
        arr = df[num_cols].apply(pd.to_numeric, errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan)
        if fit:
            med = arr.median()
        else:
            med = medians
        arr = arr.fillna(med)
        return arr, med

    Xnum_tr, medians = _clean_numeric(Xtr, fit=True)
    Xnum_va, _ = _clean_numeric(Xva, fit=False, medians=medians)
    Xnum_te, _ = _clean_numeric(Xte, fit=False, medians=medians)

    # drop zero-variance columns if requested
    removed_zero_var = []
    used_cols = num_cols.copy()
    if drop_zero_variance:
        var = Xnum_tr.var(axis=0)
        zero_cols = var[var == 0].index.tolist()
        if zero_cols:
            removed_zero_var.extend(zero_cols)
            used_cols = [c for c in used_cols if c not in zero_cols]
            Xnum_tr = Xnum_tr[used_cols]
            if Xnum_va is not None: Xnum_va = Xnum_va[used_cols]
            if Xnum_te is not None: Xnum_te = Xnum_te[used_cols]

    if not used_cols:
        f_series = pd.Series(dtype=float, index=X_train.columns)
        kept = []
        removed = X_train.columns.tolist()
        return X_train.copy(), (None if X_val is None else X_val.copy()), (None if X_test is None else X_test.copy()), kept, f_series, removed
        
    # compute ANOVA F-scores
    f_vals, _ = f_classif(Xnum_tr.values, y_train)
    f_used = pd.Series(f_vals, index=used_cols)

    # build full f_series with NaN for unused cols
    f_series = pd.Series(np.nan, index=X_train.columns, dtype="float64")
    f_series.loc[used_cols] = f_used

    # select top-k from used_cols
    k_eff = min(k, len(used_cols))
    kept_used = f_used.sort_values(ascending=False).head(k_eff).index.tolist()

    # preserve original column order
    kept = [c for c in X_train.columns if c in kept_used]
    removed = [c for c in X_train.columns if c not in kept]

    # reindex all splits
    Xtr2 = None if Xtr is None else Xtr.reindex(columns=kept, fill_value=0)
    Xva2 = None if Xva is None else Xva.reindex(columns=kept, fill_value=0)
    Xte2 = None if Xte is None else Xte.reindex(columns=kept, fill_value=0)

    logger.save_log(f"Preprocessing - ANOVA F-score feature selection: Selected top {k_eff} features by ANOVA F-score: {kept_used}, dropped zero-variance columns: {removed_zero_var}")

    return Xtr2, Xva2, Xte2, kept, f_series, removed

# PCA dimensionality reduction
def pca_reduction(
    X_train, X_val=None, X_test=None,
    n_components=0.95, random_state=42, whiten=False,
    component_prefix: str = "PC"
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # keep only numeric/bool columns
    used_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    if not used_cols:
        # Nothing numeric to run PCA on — return as-is and pca=None
        return (X_train.copy(),
                None if X_val is None else X_val.copy(),
                None if X_test is None else X_test.copy(),
                None)

    # clean numeric data: convert to numeric, coerce errors, handle inf, impute NaN with mean from train
    def _clean_numeric(df, fit=False, means=None):
        if df is None: return None, means
        arr = df[used_cols].apply(pd.to_numeric, errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan)
        if fit:
            mu = arr.mean()
        else:
            mu = means
        arr = arr.fillna(mu)
        return arr, mu

    Xnum_tr, means = _clean_numeric(Xtr, fit=True)
    Xnum_va, _ = _clean_numeric(Xva, fit=False, means=means)
    Xnum_te, _ = _clean_numeric(Xte, fit=False, means=means)

    # fit PCA
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    Xtr_p = pca.fit_transform(Xnum_tr)
    # transform val/test
    Xva_p = None if Xnum_va is None else pca.transform(Xnum_va)
    Xte_p = None if Xnum_te is None else pca.transform(Xnum_te)

    # wrap outputs
    n_comp = Xtr_p.shape[1]
    pc_cols = [f"{component_prefix}{i+1}" for i in range(n_comp)]
    Xtr_df = pd.DataFrame(Xtr_p, index=X_train.index, columns=pc_cols)
    Xva_df = None if X_val is None else pd.DataFrame(Xva_p, index=X_val.index, columns=pc_cols)
    Xte_df = None if X_test is None else pd.DataFrame(Xte_p, index=X_test.index, columns=pc_cols)

    logger.save_log(f"Preprocessing - PCA reduction: Reduced to {n_comp} principal components using PCA")

    return Xtr_df, Xva_df, Xte_df, pca


### scaling, normalization ###########

# z-score scaling
def zscore_scale_from_train(
    X_train, X_val=None, X_test=None
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # identify numeric/bool columns to scale
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    if not num_cols:
        # nothing to scale
        return Xtr, Xva, Xte, (pd.Series(dtype=float), pd.Series(dtype=float))

    # clean numeric block
    def _clean_numeric(df, fit=False, means=None):
        if df is None: return None, means
        arr = df[num_cols].apply(pd.to_numeric, errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan)
        if fit:
            mu_local = arr.mean()
        else:
            mu_local = means
        arr = arr.fillna(mu_local)
        return arr, mu_local

    Xnum_tr, mu = _clean_numeric(Xtr, fit=True)
    Xnum_va, _ = _clean_numeric(Xva, fit=False, means=mu)
    Xnum_te, _ = _clean_numeric(Xte, fit=False, means=mu)

    # standard deviations on TRAIN numeric
    sd = Xnum_tr.std().replace(0, 1.0)

    # scale numeric parts
    Xnum_tr = (Xnum_tr - mu) / sd
    if Xnum_va is not None: Xnum_va = (Xnum_va - mu) / sd
    if Xnum_te is not None: Xnum_te = (Xnum_te - mu) / sd

    # merge back into original frames (preserve order)
    def _merge(df_orig, df_num_scaled):
        if df_orig is None: return None
        out = df_orig.copy()
        out[num_cols] = df_num_scaled
        return out

    Xtr_scaled = _merge(X_train, Xnum_tr)
    Xva_scaled = _merge(X_val, Xnum_va) if X_val is not None else None
    Xte_scaled = _merge(X_test, Xnum_te) if X_test is not None else None

    logger.save_log(f"Preprocessing - z-score scaling: Scaled numeric columns using z-score scaling from TRAIN statistics")

    return Xtr_scaled, Xva_scaled, Xte_scaled, (mu, sd)

# robust scaling
def robust_scale_from_train(
    X_train, X_val=None, X_test=None
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # numeric/bool columns to scale
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    if not num_cols:
        return Xtr, Xva, Xte, (pd.Series(dtype=float), pd.Series(dtype=float))

    # clean numeric block
    def _clean_numeric(df, fit=False, med=None):
        if df is None: return None, med
        arr = df[num_cols].apply(pd.to_numeric, errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan)
        if fit:
            med_local = arr.median()
        else:
            med_local = med
        arr = arr.fillna(med_local)
        return arr, med_local

    Xnum_tr, med = _clean_numeric(Xtr, fit=True)
    Xnum_va, _   = _clean_numeric(Xva, fit=False, med=med)
    Xnum_te, _   = _clean_numeric(Xte, fit=False, med=med)

    # IQR on TRAIN numeric
    q1 = Xnum_tr.quantile(0.25)
    q3 = Xnum_tr.quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0)

    # scale numeric parts
    Xnum_tr = (Xnum_tr - med) / iqr
    if Xnum_va is not None: Xnum_va = (Xnum_va - med) / iqr
    if Xnum_te is not None: Xnum_te = (Xnum_te - med) / iqr

    # merge back into original frames (preserve order)
    def _merge(df_orig, df_num_scaled):
        if df_orig is None: return None
        out = df_orig.copy()
        out[num_cols] = df_num_scaled
        return out

    Xtr_scaled = _merge(X_train, Xnum_tr)
    Xva_scaled = _merge(X_val, Xnum_va) if X_val is not None else None
    Xte_scaled = _merge(X_test, Xnum_te) if X_test is not None else None

    logger.save_log(f"Preprocessing - robust scaling: Scaled numeric columns using robust scaling from TRAIN statistics")

    return Xtr_scaled, Xva_scaled, Xte_scaled, (med, iqr)

# min-max scaling
def minmax_scale_from_train(
    X_train, X_val=None, X_test=None,
    feature_range=(0.0, 1.0)
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    a, b = feature_range

    # numeric/bool columns to scale
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    if not num_cols:
        return Xtr, Xva, Xte, (pd.Series(dtype=float), pd.Series(dtype=float), feature_range)

    # clean numeric block and scale
    def _clean_numeric_minmax(df, fit=False, mins=None, maxs=None):
        if df is None: return None, mins, maxs
        arr = df[num_cols].apply(pd.to_numeric, errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan)
        if fit:
            minv = arr.min()
            # fill NaN with column min so we can compute max safely
            arr_filled = arr.fillna(minv)
            maxv = arr_filled.max()
        else:
            minv, maxv = mins, maxs
            arr_filled = arr.fillna(minv)
        # scale
        denom = (maxv - minv).replace(0, 1.0)
        scaled = (arr_filled - minv) / denom * (b - a) + a
        return scaled, minv, maxv
    # fit on TRAIN numeric block; transform all splits
    Xnum_tr, minv, maxv = _clean_numeric_minmax(Xtr, fit=True)
    Xnum_va, _, _ = _clean_numeric_minmax(Xva, fit=False, mins=minv, maxs=maxv)
    Xnum_te, _, _ = _clean_numeric_minmax(Xte, fit=False, mins=minv, maxs=maxv)

    # merge back into original frames (preserve order)
    def _merge(df_orig, df_num_scaled):
        if df_orig is None: return None
        out = df_orig.copy()
        out[num_cols] = df_num_scaled
        return out

    Xtr_scaled = _merge(X_train, Xnum_tr)
    Xva_scaled = _merge(X_val, Xnum_va) if X_val is not None else None
    Xte_scaled = _merge(X_test, Xnum_te) if X_test is not None else None

    logger.save_log(f"Preprocessing - min-max scaling: Scaled numeric columns using min-max scaling from TRAIN statistics to range {feature_range}")

    return Xtr_scaled, Xva_scaled, Xte_scaled, (minv, maxv, feature_range)

# L2 normalization
def l2_normalize_rows(
    X_train, X_val=None, X_test=None
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # identify numeric/bool columns to normalize
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    if not num_cols:
        # Nothing numeric to normalize
        return Xtr, Xva, Xte, None

    # clean numeric block
    def _num_block(df):
        if df is None: return None
        arr = df[num_cols].apply(pd.to_numeric, errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return arr

    Xnum_tr = _num_block(Xtr)
    Xnum_va = _num_block(Xva) if Xva is not None else None
    Xnum_te = _num_block(Xte) if Xte is not None else None

    # apply L2 normalization
    norm = Normalizer(norm="l2")
    Xnum_tr_n = pd.DataFrame(norm.fit_transform(Xnum_tr), index=Xtr.index, columns=num_cols)
    Xnum_va_n = (None if Xnum_va is None else
                 pd.DataFrame(norm.transform(Xnum_va), index=Xva.index, columns=num_cols))
    Xnum_te_n = (None if Xnum_te is None else
                 pd.DataFrame(norm.transform(Xnum_te), index=Xte.index, columns=num_cols))

    # merge back into original frames (preserve order)
    def _merge(df_orig, df_num_scaled):
        if df_orig is None: return None
        out = df_orig.copy()
        out[num_cols] = df_num_scaled
        return out

    Xtr_norm = _merge(Xtr, Xnum_tr_n)
    Xva_norm = _merge(Xva, Xnum_va_n) if Xva is not None else None
    Xte_norm = _merge(Xte, Xnum_te_n) if Xte is not None else None

    logger.save_log(f"Preprocessing - L2 normalization: Applied L2 row normalization to numeric columns")

    return Xtr_norm, Xva_norm, Xte_norm, norm


### outlier handling ###########

# outlier clipping
def clip_outliers_iqr(X_train, X_val=None, X_test=None, whisker=3.0):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()
    # identify numeric columns
    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    if not num:
        return Xtr, Xva, Xte, (None, None)
    # compute IQR-based bounds from train
    q1 = Xtr[num].quantile(0.25)
    q3 = Xtr[num].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - whisker * iqr
    hi = q3 + whisker * iqr
    # clip
    Xtr[num] = Xtr[num].clip(lo, hi, axis=1)
    if Xva is not None: Xva[num] = Xva[num].clip(lo, hi, axis=1)
    if Xte is not None: Xte[num] = Xte[num].clip(lo, hi, axis=1)

    logger.save_log(f"Preprocessing - outlier clipping: Clipped outliers using IQR method with whisker={whisker}")  

    return Xtr, Xva, Xte, (lo, hi)

def clip_by_percentile_from_train(X_train, X_val=None, X_test=None, lo=0.5, hi=99.5):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()
    # identify numeric columns
    num = Xtr.select_dtypes(include=[np.number]).columns
    # compute percentiles from train
    lows = Xtr[num].quantile(lo/100.0)
    highs = Xtr[num].quantile(hi/100.0)
    # clip
    Xtr[num] = Xtr[num].clip(lows, highs, axis=1)
    if Xva is not None: Xva[num] = Xva[num].clip(lows, highs, axis=1)
    if Xte is not None: Xte[num] = Xte[num].clip(lows, highs, axis=1)

    logger.save_log(f"Preprocessing - percentile clipping: Clipped outliers using percentiles {lo}-{hi}")   

    return Xtr, Xva, Xte, (lows, highs)

# log1p transform
def log1p_transform(X_train, X_val=None, X_test=None):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()
    # identify numeric columns with min >=0 and skew > 1.0
    num = Xtr.select_dtypes(include=[np.number]).columns
    cols = [c for c in num if (Xtr[c].min() >= 0) and (Xtr[c].skew() > 1.0)]
    # apply log1p
    for c in cols:
        Xtr[c] = np.log1p(Xtr[c])
        if Xva is not None: Xva[c] = np.log1p(Xva[c])
        if Xte is not None: Xte[c] = np.log1p(Xte[c])

    logger.save_log(f"Preprocessing - log1p transform: Applied log1p transform to columns: {cols}")

    return Xtr, Xva, Xte, cols


### class imbalance handling ###########

# class weights
def compute_class_weights(y_train, scheme='balanced'):
    # count occurrences
    cnt = Counter(y_train)
    # sorted classes
    classes = sorted(cnt.keys())
    # compute weights
    total = sum(cnt.values())
    cw = {}
    if scheme == 'balanced':
        for c in classes:
            cw[c] = total / (len(classes) * cnt[c])
    elif scheme == 'uniform':
        for c in classes:
            cw[c] = 1.0
    # compute special pos-weight for binary classification
    spw = None
    if len(classes) == 2:
        neg = cnt[classes[0]]
        pos = cnt[classes[1]]
        spw = neg / max(pos, 1)

    logger.save_log(f"Preprocessing - class weights: Computed class weights using scheme '{scheme}': {cw}, special pos-weight: {spw}")

    return cw, spw

def expand_sample_weights(y, class_weights):
    return np.array([class_weights[yi] for yi in y], dtype=float)

# resampling methods
# SMOTE
def balance_classes_smote(
    X_train, y_train,
    random_state=42, k_neighbors=5
):
    Xtr = X_train.copy()
    ytr = y_train.copy()

    # select numeric/bool columns
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    # check availability
    if not num_cols:
        raise ValueError("No numeric columns available for SMOTE. Encode categoricals first!")
    # extract numeric block
    Xnum = Xtr[num_cols]
    # apply SMOTE
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    Xnum_res, y_res = sm.fit_resample(Xnum, ytr)

    # reconstruct full X_res
    ignored_cols = [c for c in Xtr.columns if c not in num_cols]
    if ignored_cols:
        #repeat ignored columns
        reps = int(len(y_res) / len(ytr))
        ignored_repeated = pd.concat([Xtr[ignored_cols]] * reps, ignore_index=True)
        X_res = pd.concat([Xnum_res.reset_index(drop=True), ignored_repeated], axis=1)
    else:
        X_res = Xnum_res
    
    logger.save_log(f"Preprocessing - SMOTE resampling: Applied SMOTE to balance classes. Original size: {len(ytr)}, Resampled size: {len(y_res)}")

    return X_res, y_res

# random over
def balance_classes_random_over(X_train, y_train, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    logger.save_log(f"Preprocessing - random oversampling: Applied random oversampling to balance classes.")
    return ros.fit_resample(X_train, y_train)

# random under
def balance_classes_random_under(X_train, y_train, random_state=42):
    rus = RandomUnderSampler(random_state=random_state)
    logger.save_log(f"Preprocessing - random undersampling: Applied random undersampling to balance classes.")
    return rus.fit_resample(X_train, y_train)


### svm kernel approximation ###########

# Nystroem RBF kernel features
def nystroem_rbf_from_train(
    X_train, X_val=None, X_test=None,
    gamma='scale', n_components=1000, random_state=42,
    prefix: str = "RBF_NYS_"
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # Select numeric block for kernel features
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    non_num_cols = [c for c in Xtr.columns if c not in num_cols]

    if not num_cols:
        # Nothing numeric to approximate; return originals and a fitted-None
        return Xtr, Xva, Xte, None

    # Clean numeric blocks (coerce -> replace inf -> fill NaN with 0)
    def _num_block(df):
        if df is None: return None
        arr = df[num_cols].apply(pd.to_numeric, errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return arr

    Xnum_tr = _num_block(Xtr)
    Xnum_va = _num_block(Xva) if Xva is not None else None
    Xnum_te = _num_block(Xte) if Xte is not None else None

    # Gamma handling on TRAIN numeric block
    if gamma == 'scale':
        var = Xnum_tr.values.var()
        var = var if var > 0 else 1.0
        gamma_val = 1.0 / (Xnum_tr.shape[1] * var)
    elif gamma == 'auto':
        gamma_val = 1.0 / Xnum_tr.shape[1]
    else:
        gamma_val = float(gamma)

    # Fit/transform Nystroem on numeric block
    ny = Nystroem(kernel='rbf', gamma=gamma_val, n_components=n_components, random_state=random_state)
    Xtr_feat = ny.fit_transform(Xnum_tr)
    def _tx(arr):
        if arr is None: return None
        return ny.transform(arr)

    Xva_feat = _tx(Xnum_va)
    Xte_feat = _tx(Xnum_te)

    # Wrap kernel features as DataFrames with readable names
    feat_cols = [f"{prefix}{i+1}" for i in range(Xtr_feat.shape[1])]
    Xtr_feat_df = pd.DataFrame(Xtr_feat, index=X_train.index, columns=feat_cols)
    Xva_feat_df = None if X_val is None else pd.DataFrame(Xva_feat, index=X_val.index, columns=feat_cols)
    Xte_feat_df = None if X_test is None else pd.DataFrame(Xte_feat, index=X_test.index, columns=feat_cols)

    # Concatenate back non-numeric columns (untouched), preserving original order after the new features
    def _concat_with_nonnum(df_orig, feat_df):
        if df_orig is None: return None
        if not non_num_cols:
            return feat_df
        return pd.concat([feat_df, df_orig[non_num_cols].copy()], axis=1)

    Xtr_out = _concat_with_nonnum(Xtr, Xtr_feat_df)
    Xva_out = _concat_with_nonnum(Xva, Xva_feat_df) if Xva_feat_df is not None else None
    Xte_out = _concat_with_nonnum(Xte, Xte_feat_df) if Xte_feat_df is not None else None

    return Xtr_out, Xva_out, Xte_out, ny

# RBF Random Fourier Features
def rbf_sampler_from_train(
    X_train, X_val=None, X_test=None,
    gamma='scale', n_components=1000, random_state=42,
    prefix: str = "RBF_RFF_"
):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    # Numeric block for kernel features
    num_cols = Xtr.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    non_num_cols = [c for c in Xtr.columns if c not in num_cols]
    if not num_cols:
        # Nothing numeric to map
        return Xtr, Xva, Xte, None

    def _num_block(df):
        if df is None: return None
        arr = df[num_cols].apply(pd.to_numeric, errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return arr

    Xnum_tr = _num_block(Xtr)
    Xnum_va = _num_block(Xva) if Xva is not None else None
    Xnum_te = _num_block(Xte) if Xte is not None else None

    # Gamma handling on TRAIN numeric block
    if gamma == 'scale':
        var = Xnum_tr.values.var()
        var = var if var > 0 else 1.0
        gamma_val = 1.0 / (Xnum_tr.shape[1] * var)
    elif gamma == 'auto':
        gamma_val = 1.0 / Xnum_tr.shape[1]
    else:
        gamma_val = float(gamma)

    # Fit/transform RFF on numeric block
    rbf = RBFSampler(gamma=gamma_val, n_components=n_components, random_state=random_state)
    Xtr_feat = rbf.fit_transform(Xnum_tr)

    def _tx(arr):
        if arr is None: return None
        return rbf.transform(arr)

    Xva_feat = _tx(Xnum_va)
    Xte_feat = _tx(Xnum_te)

    # Wrap as DataFrames with readable names
    feat_cols = [f"{prefix}{i+1}" for i in range(Xtr_feat.shape[1])]
    Xtr_feat_df = pd.DataFrame(Xtr_feat, index=X_train.index, columns=feat_cols)
    Xva_feat_df = None if X_val is None else pd.DataFrame(Xva_feat, index=X_val.index, columns=feat_cols)
    Xte_feat_df = None if X_test is None else pd.DataFrame(Xte_feat, index=X_test.index, columns=feat_cols)

    # Concatenate back non-numeric columns (untouched), preserving original order after the new features
    def _concat_with_nonnum(df_orig, feat_df):
        if df_orig is None: return None
        if not non_num_cols:
            return feat_df
        return pd.concat([feat_df, df_orig[non_num_cols].copy()], axis=1)

    Xtr_out = _concat_with_nonnum(Xtr, Xtr_feat_df)
    Xva_out = _concat_with_nonnum(Xva, Xva_feat_df) if Xva_feat_df is not None else None
    Xte_out = _concat_with_nonnum(Xte, Xte_feat_df) if Xte_feat_df is not None else None

    return Xtr_out, Xva_out, Xte_out, rbf


### time features ###########

# add time-based features

def add_time_features(X_train, X_val=None, X_test=None, tz=None):
    Xtr = X_train.copy()
    Xva = None if X_val is None else X_val.copy()
    Xte = None if X_test is None else X_test.copy()

    def _add(df):
        if df is None or st.session_state._timeStampCol not in df.columns:
            return df
        ts = pd.to_datetime(df[st.session_state._timeStampCol], errors='coerce', utc=True)
        if tz is not None:
            try:
                ts = ts.dt.tz_convert(tz)
            except Exception:
                pass
        df = df.copy()
        df['hour'] = ts.dt.hour
        df['dow'] = ts.dt.dayofweek
        df['month'] = ts.dt.month
        df['year'] = ts.dt.year
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        return df

    return _add(Xtr), _add(Xva), _add(Xte)


# Preprocess Model
def preprocessModel(X_train, y_train, X_val, y_val, X_test, y_test, config):
    meta = {"steps": []}

    # local helpers
    def flag(k, deafult=False): return config[k] if k in config else deafult
    def sval(k, default=None): return config[k] if k in config else default

    #ss = st.session_state
    #def flag(k, default=False): return bool(ss.get(k, default))
    #def sval(k, default=None):   return ss.get(k, default)

    # copies
    Xtr = X_train.copy()
    ytr = pd.Series(y_train).copy()
    Xva = None if X_val is None else X_val.copy()
    yva = None if y_val is None else pd.Series(y_val).copy()
    Xte = None if X_test is None else X_test.copy()
    yte = None if y_test is None else pd.Series(y_test).copy()

    # 0) Encode labels if not numeric (fit on y_train only)
    y_enc_map = None
    if not np.issubdtype(pd.Series(ytr).dtype, np.number):
        ytr, yva, yte, y_enc_map = encode_labels(ytr, yva, yte)
        meta["label_encoding"] = {"mapping": y_enc_map}
        meta["steps"].append(("encode_labels", {}))

    # 1) Time features (no fit; safe)
    if flag("_TF"):
        Xtr, Xva, Xte = add_time_features(
            Xtr, Xva, Xte, tz=None
        )
        meta["steps"].append(("add_time_features", {"ts_col": "timestamp"}))

    # 2) Drop duplicates (keeps labels aligned)
    if flag("_DD"):
        Xtr, ytr, Xva, yva, Xte, yte, stats= drop_duplicates(Xtr, ytr, Xva, yva, Xte, yte)
        meta["steps"].append(("drop_duplicates", {}))
        meta["drop duplicates results"]= {"stats": stats}

    # 3) Missing values (fit on train)
    if flag("_MV"):
        num_strat = "median" if sval("_MV_N", "Median") == "Median" else "most_frequent"
        cat_strat = "most_frequent"
        Xtr, Xva, Xte, impn, impc = impute_numeric_categorical(
            Xtr, Xva, Xte, num_strategy=num_strat, cat_strategy=cat_strat
        )
        meta["steps"].append(("impute_numeric_categorical", {"num": num_strat, "cat": cat_strat}))

    # 4) Bucket rare categories BEFORE encodings
    if flag("_BR"):
        min_freq = int(sval("_BR_MF", 20))
        Xtr, Xva, Xte, rare_maps = bucket_rare_categories(Xtr, Xva, Xte, min_freq=min_freq)
        meta["steps"].append(("bucket_rare_categories", {"min_freq": min_freq}))
        meta["rare_bucket_maps"] = {c: len(v) for c, v in rare_maps.items()}

    # 5) Encoding
    used_encoding = None
    if flag("_GBM"):
        lib = sval("_GBM_M", "Light GBM")
        if lib == "CatBoost":
            Xtr, Xva, Xte, cat_idx = prepare_categoricals_gbm(Xtr, Xva, Xte, "catboost")
            meta["steps"].append(("prepare_categoricals_for_catboost", {}))
            meta["gbm_cat_features_idx"] = cat_idx
            used_encoding = "gbm_catboost"
        elif lib == "Light GBM":
            Xtr, Xva, Xte, cat_cols = prepare_categoricals_gbm(Xtr, Xva, Xte, "lightgbm")
            meta["steps"].append(("prepare_categoricals_for_lightgbm", {}))
            meta["gbm_cat_cols"] = cat_cols
            used_encoding = "gbm_lightgbm"
        else:  # "XG Boost"
            Xtr, Xva, Xte, cat_cols = prepare_categoricals_gbm(Xtr, Xva, Xte, "xgboost")
            meta["steps"].append(("prepare_categoricals_for_xgboost", {}))
            meta["gbm_cat_cols"] = cat_cols
            used_encoding = "gbm_xgboost"

    if not flag("_GBM") and flag("_EN_GA"):
        if sval("_EN_GA_O", "One Hot Encoding") == "One Hot Encoding":
            Xtr, Xva, Xte, ohe = encode_one_hot(Xtr, Xva, Xte, handle_unknown="ignore")
            meta["steps"].append(("encode_one_hot", {"handle_unknown": "ignore"}))
            used_encoding = "one_hot"
        else:
            Xtr, Xva, Xte, ord_enc = encode_ordinal(Xtr, Xva, Xte, categories="auto")
            meta["steps"].append(("encode_ordinal", {"categories": "auto"}))
            used_encoding = "ordinal"

    # 6) Extra encodings that add features
    if flag("_CE"):
        Xtr, Xva, Xte, cols, maps = count_encoding(X_train, Xtr, Xva, Xte)
        meta["steps"].append(("count_encoding", {"cols": cols}))
        meta["count_maps"] = {k: len(v) for k, v in maps.items()} if cols else {}

    if flag("_TME"):
        n_splits = int(sval("_TME_NS", 5))
        noise_std = float(sval("_TMEN_NOS", 0.0))
        smooth = float(sval("_TME_GS", 10.0))
        Xtr, Xva, Xte, enc_maps = target_mean_encoding_kfold(
            Xtr, ytr, Xva, Xte,
            cols=None, n_splits=n_splits, noise_std=noise_std, global_smoothing=smooth
        )
        meta["steps"].append(("target_mean_encoding_kfold", {
            "n_splits": n_splits, "noise_std": noise_std, "global_smoothing": smooth
        }))
        meta["tgt_mean_maps"] = {k: len(v[0]) for k, v in enc_maps.items()}

    # 7) Feature pruning / selection
    if flag("_DC"):
        thr = float(sval("_DC_V", 0.0))
        Xtr, Xva, Xte, kept, removed = drop_low_variance(Xtr, Xva, Xte, threshold=thr)
        meta["steps"].append(("drop_low_variance", {"threshold": thr}))
        meta["kept_after_low_variance"] = kept
        meta["dropped_after_low_variance"] = removed

    if flag("_DHC"):
        thr = float(sval("_DHC_T", 0.98))
        Xtr, Xva, Xte, kept, dropped = drop_high_corr_numeric(Xtr, Xva, Xte, threshold=thr)
        meta["steps"].append(("drop_high_corr_numeric", {"threshold": thr}))
        meta["dropped_after_high_corr"] = dropped

    if flag("_FS_RF"):
        n_keep = int(sval("_FS_RF_TN", 100))
        Xtr, Xva, Xte, kept, importances, removed = select_features_by_rf_importance(
            Xtr, ytr, X_val=Xva, X_test=Xte, n_keep=n_keep
        )
        meta["steps"].append(("select_features_by_rf_importance", {"n_keep": n_keep}))
        meta["rf_kept"] = kept
        meta["rf_importances"] = importances
        meta["rf_removed"] = removed

    if flag("_FS_A"):
        k = int(sval("_FS_A_TK", 200))
        Xtr, Xva, Xte, kept, f_scores, removed = select_by_anova_f(Xtr, ytr, X_val=Xva, X_test=Xte, k=k)
        meta["steps"].append(("select_by_anova_f", {"k": k}))
        meta["anova_kept"] = kept
        meta["anova_removed"] = removed
        meta["anova_f_scores"] = f_scores
    
    if flag("_FS_MI"):
        k = int(sval("_FS_MI_TK", 200))
        Xtr, Xva, Xte, kept, mi_scores, removed = select_by_mutual_info(Xtr, ytr, X_val=Xva, X_test=Xte, k=k)
        meta["steps"].append(("select_by_mutual_info", {"k": k}))
        meta["mi_kept"] = kept
        meta["mi_removed"] = removed
        meta["mi_scores"] = mi_scores

    if flag("_PCA"):
        n_components = sval("_PCA_N", 0.95)
        Xtr, Xva, Xte, pca = pca_reduction(Xtr, Xva, Xte, n_components=n_components)
        meta["steps"].append(("pca_reduction", {"n_components": n_components}))

    # 8) Outliers & transforms (BEFORE scaling)
    if flag("_OC"):
        method = sval("_OC_M", "IQR")
        if method == "IQR":
            w = float(sval("_OC_WW", 3.0))
            Xtr, Xva, Xte, bounds = clip_outliers_iqr(Xtr, Xva, Xte, whisker=w)
            meta["steps"].append(("clip_outliers_iqr", {"whisker": w}))
        else:
            lo = float(sval("_OC_PL", 0.5)); hi = float(sval("_OC_M_PH", 99.5))
            Xtr, Xva, Xte, bounds = clip_by_percentile_from_train(Xtr, Xva, Xte, lo=lo, hi=hi)
            meta["steps"].append(("clip_by_percentile_from_train", {"lo": lo, "hi": hi}))

    if flag("_LOG"):
        Xtr, Xva, Xte, log_cols = log1p_transform(Xtr, Xva, Xte)
        meta["steps"].append(("log1p_transform_from_train", {"cols": log_cols}))

    # 9) Scaling / normalization (AFTER clipping/log)
    if flag("_SC"):
        scaler = sval("_SC_S", "Z-Score")
        if scaler == "Z-Score":
            Xtr, Xva, Xte, stats = zscore_scale_from_train(Xtr, Xva, Xte)
            meta["steps"].append(("zscore_scale_from_train", {}))
        elif scaler == "Robust":
            Xtr, Xva, Xte, stats = robust_scale_from_train(Xtr, Xva, Xte)
            meta["steps"].append(("robust_scale_from_train", {}))
        else:  # MinMax
            a = float(sval("_SC_S_MIN", 0.0))
            b = float(sval("_SC_S_MAX", 1.0))
            Xtr, Xva, Xte, stats = minmax_scale_from_train(Xtr, Xva, Xte, feature_range=(a, b))
            meta["steps"].append(("minmax_scale_from_train", {"range": (a, b)}))

    if flag("_L2N"):
        Xtr, Xva, Xte, norm = l2_normalize_rows(Xtr, Xva, Xte)
        meta["steps"].append(("l2_normalize_rows", {}))

    # 10) Kernel approximations (AFTER scaling)
    if flag("_SVM"):
        method = sval("_SVM_M", "Nyström")
        if method == "Nyström":
            n_comp = int(sval("_SVM_NNC", 2000))
            Xtr, Xva, Xte, ny = nystroem_rbf_from_train(Xtr, Xva, Xte, gamma="scale", n_components=n_comp)
            meta["steps"].append(("nystroem_rbf_from_train", {"n_components": n_comp}))
        else:
            n_comp = int(sval("_SVM_RFFNC", 2000))
            Xtr, Xva, Xte, rff = rbf_sampler_from_train(Xtr, Xva, Xte, gamma="scale", n_components=n_comp)
            meta["steps"].append(("rbf_sampler_from_train", {"n_components": n_comp}))

    # 11) Class imbalance
    class_weights = None
    scale_pos_weight = None
    sample_weight_train = None
    sample_weight_val = None
    sample_weight_test = None

    if flag("_CW"):
        cw_scheme = sval("_CW_S", "balanced")
        class_weights, scale_pos_weight = compute_class_weights(ytr, scheme=cw_scheme)
        meta["steps"].append(("compute_class_weights", {"scheme": cw_scheme}))
        meta["class_weights"] = class_weights
        meta["scale_pos_weight"] = scale_pos_weight

        sample_weight_train = expand_sample_weights(ytr, class_weights)
        if yva is not None:
            sample_weight_val = expand_sample_weights(yva, class_weights)
        if yte is not None:
            sample_weight_test = expand_sample_weights(yte, class_weights)
        meta["steps"].append(("expand_sample_weights", {}))


    if flag("_RS"):
        method = sval("_RS_M", "SMOTE")
        if method == "SMOTE":
            k = int(sval("_RS_SK", 5))
            Xtr, ytr = balance_classes_smote(Xtr, ytr, k_neighbors=k)
            meta["steps"].append(("balance_classes_smote", {"k_neighbors": k}))
        elif method == "Random Over":
            Xtr, ytr = balance_classes_random_over(Xtr, ytr)
            meta["steps"].append(("balance_classes_random_over", {}))
        else:
            Xtr, ytr = balance_classes_random_under(Xtr, ytr)
            meta["steps"].append(("balance_classes_random_under", {}))

        # If we resampled, per-sample weights no longer match; recompute if class weights are ON
        if class_weights is not None:
            sample_weight_train = expand_sample_weights(ytr, class_weights)
            meta["steps"].append(("expand_sample_weights_after_resample", {}))

    # 12) Final alignment (safety) — ensure same columns in val/test as train

    Xva = None if Xva is None else Xva.reindex(columns=Xtr.columns, fill_value=0)
    Xte = None if Xte is None else Xte.reindex(columns=Xtr.columns, fill_value=0)

    meta["final_shapes"] = {
        "X_train": tuple(Xtr.shape),
        "X_val": None if Xva is None else tuple(Xva.shape),
        "X_test": None if Xte is None else tuple(Xte.shape),
    }

    return {
        "X_train": Xtr, 
        "y_train": ytr,
        "X_val":   Xva, 
        "y_val":   yva,
        "X_test":  Xte, 
        "y_test":  yte,
        "class_weights": class_weights,
        "scale_pos_weight": scale_pos_weight,
        "sample_weight_train": sample_weight_train,
        "sample_weight_val": sample_weight_val,
        "sample_weight_test": sample_weight_test,
        "metadata": meta,
        "used_encoding": used_encoding,
    }