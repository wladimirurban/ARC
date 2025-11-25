import streamlit as st

import functions.f00_Logger as logger

import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

################
### INSIGHTS ###
################

def analyze_sparsity(df):
    # all cells
    total_cells = df.size
    # missing cells
    total_missing = df.isnull().sum().sum()
    # missing percent
    total_missing_percent = round((total_missing / total_cells) * 100, 2)
    # missing percent per column
    col_missing = df.isnull().mean().round(4) * 100
    # filter only columns with missing values
    col_missing = col_missing[col_missing > 0].sort_values(ascending=False)
    # return results
    return{
        "Total missing cells": total_missing,
        "Total missing percent": total_missing_percent, 
        "Col missing": col_missing
    }

def detect_granularity(df):
    # normalize column names for easier matching
    cols_norm = [normalize_column_name(c) for c in df.columns]
    # faster membership checks
    cols_set = set(cols_norm) 

    # normalized targets
    flow_keys     = {"flowid", "flowlabel"}
    packet_exact  = {"framenumber", "framelen"}
    session_keys  = {"session", "sessionid"}

    # flow-level
    if any(k in cols_set for k in flow_keys):
        return "Flow-level"
    # packet-level
    if any(c.startswith("packet") for c in cols_norm) or any(k in cols_set for k in packet_exact):
        return "Packet-level"
    # session-level
    if any(k in cols_set for k in session_keys):
        return "Session-level"
    # unknown / custom
    return "Unknown / Custom"
    
def analyze_feature_set(df):
    # get columns by data type
    numerical = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['category']).columns.tolist()
    bool = df.select_dtypes(include=['bool']).columns.tolist()
    object = df.select_dtypes(include=['object']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime']).columns.tolist()
    # return summary
    return {
        "Total Features": len(df.columns),
        "Numerical": numerical,
        "Categorical": categorical,
        "Bool": bool,
        "Object": object,
        "Datetime": datetime
    }

def analyze_time_span(df, timestamp_col):
    try:
        # convert to datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        # get valid dates only
        valid_dates = df[timestamp_col].dropna().sort_values()
        # not enough valid dates
        if len(valid_dates) < 1:
            return {
                "Start": None, 
                "End": None, 
                "Duration": None,
                "Most common interval": None,
                "Average interval": None,
                "Exception": "Not enough entries"
            }
        # analyze time span
        start = valid_dates.min()
        end = valid_dates.max()
        duration = end - start
        deltas = valid_dates.diff().dropna()
        most_common_delta = deltas.mode()[0] if not deltas.mode().empty else None
        average_delta = deltas.mean()
        # return results
        return {
            "Start": start, 
            "End": end, 
            "Duration": duration, 
            "Most common interval": most_common_delta,
            "Average interval": average_delta,
            "Exception": None
        }
    # catch exceptions
    except Exception as e:
        e = (f"Could not analyze time span: {e}")
        return {
            "Start": None, 
            "End": None, 
            "Duration": None, 
            "Most common interval": None,
            "Average interval": None,
            "Exception": e
        }

def check_column_presence(df, required_columns):
    # list current columns
    current_cols = df.columns.tolist()
    # check for missing and extra columns
    missing_cols = [col for col in required_columns if col not in current_cols]
    extra_cols = [col for col in current_cols if col not in required_columns]
    # return results
    return{
        "Missing required columns": missing_cols,
        "Extra/unexpected columns": extra_cols
    }

###############
### HELPERS ###
###############

def parse_rename_matrix(text):
    rename_dict = {}
    # parse lines
    lines = text.strip().splitlines()
    for line in lines:
        # skip empty lines
        if "=" in line:
            # split key and value
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                rename_dict[key] = value
    return rename_dict

####################
### MODIFICATION ###
####################

def normalize_column_names(df):
    # make a copy to avoid modifying original
    df = df.copy()
    # normalize column names
    new_columns = [normalize_column_name(col) for col in df.columns]
    df.columns = new_columns
    # log
    logger.save_log("Features normalized")
    st.session_state._SV_NormCol = True
    return df

def normalize_column_name(col):
    # normalize by lowercasing, stripping spaces, removing internal spaces
    return col.lower().strip().replace(" ", "") if col else col

def rename_column(col_name, rename_map):
    # rename single column
    if isinstance(rename_map, str):
        rename_map = parse_rename_matrix(rename_map) or {}

    return rename_map.get(col_name, col_name)

def rename_columns(df, rename_map):
    # rename multiple columns
    if isinstance(rename_map, str):
        rename_map = parse_rename_matrix(rename_map) or {}

    # make a copy to avoid modifying original
    #df = df.copy()

    # build rename dict
    rename_dict = {col: rename_map[col] for col in df.columns if col in rename_map}
    # apply renaming
    if rename_dict:
        df = df.rename(columns=rename_dict)

        # log
        st.session_state._SV_RenameCol = True
        for k, v in rename_dict.items():
            logger.save_log(f"Feature {k} renamed to {v}")
            st.session_state._SV_RenameCols.append((f"Feature {k} renamed to {v}"))

    return df

def sort_by_timestamp(df, timestamp_col, add_delta = False):
    # make a copy to avoid modifying original
    df = df.copy()

    # convert to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    # sort by timestamp
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)

    # log
    st.session_state._SV_SortByT = True
    logger.save_log("Dataset sorted by timestamp")

    # add delta column if requested (do NOT overwrite df)
    if add_delta:
        # create a new column with seconds difference between consecutive timestamps
        df["delta_seconds"] = df[timestamp_col].diff().dt.total_seconds().fillna(0)
        # log
        st.session_state._SV_DeltaT = True
        logger.save_log("Delta time column 'delta_seconds' added")

    return df

def drop_column(df, col):
    # make a copy to avoid modifying original
    df = df.copy()
    # drop specified columns if they exist
    if col in df.columns:
        df = df.drop(columns=col)
        # log
        st.session_state._SV_DropCol = True
        st.session_state._SV_DropedCols.append(col)
        logger.save_log("Feature " + str(col) + " dropped")
    
    return df

def drop_duplicates(df):
    # drop duplicate rows
    before_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after_count = len(df)
    
    # log
    st.session_state._SV_DropDup = True
    st.session_state._SV_DropedDupCount = before_count - after_count
    logger.save_log(f"Dropped {before_count - after_count} duplicate rows.")
    
    return df

##############
### RUNNER ###
##############

def schemaValidatorTotal(df, hastimestamp, timestamp_col):
    total_records = len(df)
    duplicates = df.duplicated().sum()
    sparsity = analyze_sparsity(df)
    granularity = detect_granularity(df)
    features_unique = len(df.columns) != len(set(df.columns))
    # detect mixed type columns
    mixed_type_cols = []
    for col in df.columns:
        types = df[col].dropna().map(type).nunique()
        if types > 1:
            mixed_type_cols.append(col) 
    features = analyze_feature_set(df)
    if hastimestamp:
        time_dup = df[timestamp_col].duplicated().any()
        time_sorted = df[timestamp_col].is_monotonic_increasing
        time_analysis = analyze_time_span(df, timestamp_col)
    else:
        time_dup = None
        time_sorted = None
        time_analysis = None

    return{
        "total_records": total_records,
        "duplicates": duplicates,
        "sparsity": sparsity,  
        "granularity": granularity,
        "features_unique": features_unique,   
        "mixed_type_cols": mixed_type_cols,
        "features": features,
        "time_dup": time_dup,
        "time_sorted": time_sorted,
        "time_analysis": time_analysis
    }
