import streamlit as st

import functions.f00_Logger as logger

import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px

################
### Splitter ###
################

def split_dataset(df, label_col, time_col, method, test_size: float, val_size: float, gap_ratio: float = 0.0, random_state: int = 42):
    # basic checks
    total = test_size + val_size + (gap_ratio if method == "Time-based Split" else 0.0)
    if not (0 < test_size < 1 and 0 < val_size < 1 and total < 1):
        st.error("Invalid split fractions. Ensure 0 < val,test < 1 and (val+test [+gap]) < 1.")
        return

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # -------------------------
    # RANDOM / STRATIFIED cases
    # -------------------------
    if method in ("Random Split", "Stratified Split"):
        strat = (y if method == "Stratified Split" else None)

        # First split: Train vs Temp (Val+Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(test_size + val_size),
            stratify=strat,
            random_state=random_state,
            shuffle=True,
        )

        # Second split: Temp -> Validation vs Test
        X_validate, X_test, y_validate, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size / (test_size + val_size),
            stratify=(y_temp if method == "Stratified Split" else None),
            random_state=random_state,
            shuffle=True,
        )
        logger.save_log(f"Dataset split using {method} (traine: {(1-val_size-test_size)*100}%, test: {test_size*100:.1f}%, val: {val_size*100:.1f}%) with random_state={random_state}")

    # -------------------------
    # TIME-BASED case (with gap)
    # -------------------------
    elif method == "Time-based Split":
        if not getattr(st.session_state, "_hasTimeStamp", False) or time_col not in df.columns:
            st.error("No valid timestamp column for time-based split.")
            return

        # sort by time (be robust to dtype)
        df_sorted = df.sort_values(by=time_col, key=lambda s: pd.to_datetime(s, errors="coerce"))

        n = len(df_sorted)
        n_gap = int(round(n * gap_ratio))
        n_train = int(round(n * (1 - (test_size + val_size + gap_ratio))))
        if n_train <= 0 or (n_train + n_gap) >= n:
            st.error("Gap/train sizes leave no room for validation/test. Adjust gap_ratio/val/test.")
            return

        # Train (earliest), then GAP (ignored), then TEMP (Val+Test)
        train_df = df_sorted.iloc[:n_train]
        temp_df  = df_sorted.iloc[n_train + n_gap:]

        # Split TEMP chronologically into Val and Test (no shuffle, no stratify)
        temp_n = len(temp_df)
        n_val = int(round(temp_n * (val_size / (val_size + test_size))))
        val_df  = temp_df.iloc[:n_val]
        test_df = temp_df.iloc[n_val:]

        X_train = train_df.drop(columns=[label_col]); y_train = train_df[label_col]
        X_validate = val_df.drop(columns=[label_col]); y_validate = val_df[label_col]
        X_test = test_df.drop(columns=[label_col]); y_test = test_df[label_col]

        logger.save_log(f"Dataset split using {method} (train: {(1-val_size-test_size-gap_ratio)*100}%, gap: {gap_ratio*100}%, test: {test_size*100:.1f}%, val: {val_size*100:.1f}%) with random_state={random_state}")

    else:
        st.error(f"Unknown split method: {method}")
        return

    st.session_state._SP_IsSplit = True
    return X_train, y_train, X_validate, y_validate, X_test, y_test

################
### Get data ###
################

def getDistributionSet(y_train, y_validate, y_test):

    dist_cols = sorted(set(y_train.unique()) | set(y_validate.unique()) | set(y_test.unique()))

    train_counts = y_train.value_counts().reindex(dist_cols).fillna(0).astype(int)
    val_counts   = y_validate.value_counts().reindex(dist_cols).fillna(0).astype(int)
    test_counts  = y_test.value_counts().reindex(dist_cols).fillna(0).astype(int)

    # Totals per split
    total_train, total_val, total_test = len(y_train), len(y_validate), len(y_test)

    dist_df = pd.DataFrame({
        "train_count": train_counts,
        "val_count":   val_counts,
        "test_count":  test_counts,
        "train_%": (train_counts / total_train * 100).round(2) if total_train > 0 else 0,
        "val_%":   (val_counts   / total_val   * 100).round(2) if total_val > 0 else 0,
        "test_%":  (test_counts  / total_test  * 100).round(2) if total_test > 0 else 0,
    })

    dist_df.index.name = "Label"
    return dist_df

def getDistributionLabel(y_train, y_validate, y_test):
    y_train = st.session_state._SP_y_Train
    y_validate = st.session_state._SP_y_Validate
    y_test = st.session_state._SP_y_Test

    dist_cols = sorted(set(y_train.unique()) | set(y_validate.unique()) | set(y_test.unique()))

    train_counts = y_train.value_counts().reindex(dist_cols).fillna(0).astype(int)
    val_counts   = y_validate.value_counts().reindex(dist_cols).fillna(0).astype(int)
    test_counts  = y_test.value_counts().reindex(dist_cols).fillna(0).astype(int)

    # Row totals (per label)
    row_totals = train_counts + val_counts + test_counts
    row_totals = row_totals.replace(0, 1) 

    dist_df = pd.DataFrame({
        "train_count": train_counts,
        "val_count":   val_counts,
        "test_count":  test_counts,
        "train_%": (train_counts / row_totals * 100).round(2),
        "val_%":   (val_counts   / row_totals * 100).round(2),
        "test_%":  (test_counts  / row_totals * 100).round(2),
    })

    dist_df.index.name = "Label"
    return dist_df

def getDistributionTotal(y_train, y_validate, y_test):
    """
    Distribution normalized by the total number of records (Train+Validate+Test).
    Percentages represent share of total dataset (sums to 100 across all sets and labels).
    """
    y_train = st.session_state._SP_y_Train
    y_validate = st.session_state._SP_y_Validate
    y_test = st.session_state._SP_y_Test

    dist_cols = sorted(set(y_train.unique()) | set(y_validate.unique()) | set(y_test.unique()))

    train_counts = y_train.value_counts().reindex(dist_cols).fillna(0).astype(int)
    val_counts   = y_validate.value_counts().reindex(dist_cols).fillna(0).astype(int)
    test_counts  = y_test.value_counts().reindex(dist_cols).fillna(0).astype(int)

    total_records = len(y_train) + len(y_validate) + len(y_test)
    if total_records == 0:
        total_records = 1  # avoid div/0

    dist_df = pd.DataFrame({
        "train_count": train_counts,
        "val_count":   val_counts,
        "test_count":  test_counts,
        "train_%": (train_counts / total_records * 100).round(2),
        "val_%":   (val_counts   / total_records * 100).round(2),
        "test_%":  (test_counts  / total_records * 100).round(2),
    })

    dist_df.index.name = "Label"
    return dist_df

def getunseen(y_train, y_validate, y_test):
    unseen_test = sorted(set(y_test.unique()) - set(y_train.unique()))
    unseen_val = sorted(set(y_validate.unique()) - set(y_train.unique()))
    return{
        "unseen_test": unseen_test,
        "unseen_val": unseen_val
    }

###############
### Plotter ###
###############

def plot_distribution_bars(dist_df: pd.DataFrame, height=300):
    pct_cols = [c for c in dist_df.columns if c.endswith("%")]
    if not pct_cols:
        raise ValueError("No percentage columns found (expected 'train_%', 'val_%', 'test_%').")

    df_plot = (
        dist_df.reset_index()
        .melt(id_vars="Label", value_vars=pct_cols, var_name="set", value_name="percentage")
    )

    df_plot["set"] = df_plot["set"].str.replace("_%", "", regex=False).str.capitalize()

    fig = px.bar(
        df_plot,
        x="percentage",
        y="set",
        color="Label",
        orientation="h",
        barmode="stack",
        text="percentage",
        category_orders={"set": ["Train", "Val", "Test"]},
        height=height
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="inside",
        insidetextanchor="middle"
    )

    fig.update_layout(
        xaxis_title="Percentage",
        yaxis_title="",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title=""
        )
    )

    return fig

def plot_distribution_label(dist_df: pd.DataFrame, height=300):
    pct_cols = [c for c in dist_df.columns if c.endswith("%")]
    if not pct_cols:
        raise ValueError("No percentage columns found (expected 'train_%', 'val_%', 'test_%').")

    df_plot = (
        dist_df.reset_index()
        .melt(id_vars="Label", value_vars=pct_cols, var_name="set", value_name="percentage")
    )

    df_plot["set"] = df_plot["set"].str.replace("_%", "", regex=False).str.capitalize()

    fig = px.bar(
        df_plot,
        x="Label",
        y="percentage",
        color="set",
        barmode="stack",
        text="percentage",
        height=height
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="inside"
    )

    fig.update_layout(
        xaxis_title="Label",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title=""
        )
    )

    return fig

def make_distribution_table(dist_df: pd.DataFrame) -> pd.DataFrame:
    if dist_df is None or dist_df.empty:
        return pd.DataFrame(columns=["Label", "Train", "Validate", "Test"])

    # If your dist_df has Label as index, bring it out
    df = dist_df.reset_index() if "Label" not in dist_df.columns else dist_df.copy()

    def fmt(count, pct):
        return f"{int(count)} ({pct:.1f}%)"

    out = pd.DataFrame({
        "Label": df["Label"].astype(str),
        "Train": [f"{int(c)} ({p:.1f}%)" for c, p in zip(df["train_count"], df["train_%"])],
        "Validate": [f"{int(c)} ({p:.1f}%)" for c, p in zip(df["val_count"], df["val_%"])],
        "Test": [f"{int(c)} ({p:.1f}%)" for c, p in zip(df["test_count"], df["test_%"])],
    })
    return out
