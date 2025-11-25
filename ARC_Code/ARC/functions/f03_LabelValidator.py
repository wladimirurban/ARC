import streamlit as st

import functions.f00_Logger as logger

import numpy as np
import pandas as pd
import plotly.express as px
import warnings

from typing import List
from scipy.stats import entropy

warnings.filterwarnings("ignore", category=UserWarning)

##############
### HELPER ###
##############

def get_rare_classes(df, label_col, input_r_c):
    # get class ratios
    ratios = df[label_col].value_counts(normalize=True)
    # compute threshold
    threshold = input_r_c / 100.0
    # get rare classes
    rare_classes = ratios[ratios < threshold]
    
    return rare_classes

def get_dominant_classes(df, label_col, input_d_c):
    # get class ratios
    ratios = df[label_col].value_counts(normalize=True)
    # compute threshold
    threshold = input_d_c / 100.0
    # get dominant classes
    dominant_classes = ratios[ratios > threshold]

    return dominant_classes

def get_timebin(df, label_col, timestamp_col, mode, n_bins):
    # validate inputs
    if n_bins is None or n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    mode = mode.lower().strip()
    if mode not in {"time", "records"}:
        raise ValueError('mode must be "time" or "records"')

    # prepare timestamp and label columns
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    valid = ts.notna() & df[label_col].notna()
    if not valid.any():
        return pd.DataFrame(columns=["interval"])

    # filter data
    df2 = df.loc[valid].copy()
    ts2 = ts.loc[valid]

    # binning
    if mode == "time":
        start, end = ts2.min(), ts2.max()
        if start == end:
            end = end + pd.Timedelta(microseconds=1)
        edges = pd.date_range(start=start, end=end, periods=n_bins + 1).unique()
        if len(edges) < 2:
            edges = pd.DatetimeIndex([start, start + pd.Timedelta(microseconds=1)])
        cat = pd.cut(ts2, bins=edges, right=False, include_lowest=True)
        categories = cat.cat.categories
    else:  # mode == "records"
        cat = pd.qcut(ts2, q=n_bins, duplicates="drop")
        if cat.isna().all():
            return pd.DataFrame(columns=["interval"])
        categories = cat.cat.categories

    # readable interval labels
    interval_labels = [
        f"{iv.left.strftime('%Y-%m-%d %H:%M:%S')} - {iv.right.strftime('%Y-%m-%d %H:%M:%S')}"
        for iv in categories
    ]
    label_map = dict(zip(categories, interval_labels))
    df2["interval"] = pd.Series(cat).map(label_map)

    # alphabetical label order
    labels_sorted: List[str] = (
        pd.Series(df2[label_col].unique())
          .dropna()
          .astype(str)
          .sort_values()
          .tolist()
    )

    # counts per (interval, label)
    counts = (
        df2.groupby(["interval", label_col]).size()
           .unstack(fill_value=0)
           .rename(columns=str)                        # make sure columns are strings
           .reindex(columns=labels_sorted, fill_value=0)
           .reindex(interval_labels, fill_value=0)     # include empty bins
    )

    # percentages (0..100) per interval
    totals = counts.sum(axis=1).replace(0, pd.NA)
    percentages = (counts.div(totals, axis=0) * 100).fillna(0.0)

    # build wide output with aligned index, then reset
    data = pd.DataFrame(index=counts.index)
    for lab in labels_sorted:
        data[f"{lab} %"] = percentages[lab].astype(float)
        data[f"{lab} count"] = counts[lab].astype(int)
    data = data.reset_index().rename(columns={"index": "interval"})

    return data

#####################
### HARRY PLOTTER ###
#####################

def plot_pie_chart(labels, values, height=300):
    df = pd.DataFrame({"label": labels, "value": values})
    df = df.sort_values("label")
    # create pie chart
    fig = px.pie(
        df,
        names="label",
        values="value",
        color="label",
        category_orders={"label": df["label"].tolist()}
    )
    # customize appearance
    fig.update_traces(
        textinfo="label+percent",
        textposition="inside",
        insidetextorientation="radial"
    )
    # layout
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True,
        height=height
    )
    return fig

def plot_bar_chart(labels, values, height=300):
    df = pd.DataFrame({"label": labels, "value": values})
    # sort by label
    df = df.sort_values("label")
    # create bar chart
    fig = px.bar(
        df,
        x="label",
        y="value",
        text="value",
        color="label",
        category_orders={"label": df["label"].tolist()}
    )
    # layout
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True,
        height=height
    )

    return fig

def plot_state_timeline(df, label_col, timestamp_col):
    s = (
        df.dropna(subset=[timestamp_col, label_col])
          .sort_values(timestamp_col)[[timestamp_col, label_col]]
    )
    # check for empty
    if s.empty:
        return None

    # build segments
    run_id = (s[label_col] != s[label_col].shift()).cumsum()
    segments = (
        s.groupby(run_id)
         .agg(start=(timestamp_col, 'min'),
              end=(timestamp_col, 'max'),
              label=(label_col, 'first'))
         .reset_index(drop=True)
    )

    # add constant lane for y-axis
    segments["lane"] = "Label"
    # determine label order
    labels_order = sorted(df[label_col].dropna().unique().tolist())

    #  create timeline plot
    fig_timeline = px.timeline(
        segments,
        x_start="start",
        x_end="end",
        y="lane",
        color="label",
        category_orders={"label": labels_order}
    )
    # customize layout
    fig_timeline.update_layout(
        showlegend=True,
        height=300,
        xaxis_title="Time",
        yaxis_title="",
        margin=dict(l=0, r=0, t=0, b=0),
        bargap=0.05,
        legend_traceorder="normal"
    )
    # hide y-axis ticks
    fig_timeline.update_yaxes(showticklabels=False)

    return fig_timeline

def plot_timebin(data, height = 300):
    # prepare data
    if data is None or data.empty or "interval" not in data.columns:
        return None

    count_cols = [c for c in data.columns if c.endswith(" count")]
    if not count_cols:
        return None
    labels = [c[:-6].strip() for c in count_cols]
    labels_order = sorted(labels)
    # build data for plotting
    counts_only = data[["interval"] + count_cols].copy()
    counts_only = counts_only.rename(columns={f"{lab} count": lab for lab in labels})
    df_plot = counts_only.melt(id_vars="interval", var_name="label", value_name="Count")
    # ensure categorical ordering
    df_plot["label"] = pd.Categorical(df_plot["label"], categories=labels_order, ordered=True)
    # create stacked bar chart
    fig = px.bar(
        df_plot,
        x="interval",
        y="Count",
        color="label",
        barmode="stack",
        height=height,
        category_orders={"label": labels_order},
    )
    # customize layout
    fig.update_layout(
        xaxis_title="Time Bin",
        yaxis_title="Count",
        legend_title=st.session_state._labelCol if "_labelCol" in st.session_state else "Label",
        margin=dict(l=0, r=0, t=0, b=0),
        legend_traceorder="normal",
    )

    return fig

def format_time_bin_table(data):
    # prepare data
    interval = data["interval"]
    labels = sorted(set(c.rsplit(" ", 1)[0] for c in data.columns if c not in ["interval"]))
    # build formatted table
    formatted = pd.DataFrame({"interval": interval})
    for lab in labels:
        pct_col = f"{lab} %"
        count_col = f"{lab} count"
        if pct_col in data.columns and count_col in data.columns:
            formatted[lab] = data.apply(
                lambda row: f"{row[count_col]} ({row[pct_col]:.1f}%)", axis=1
            )
    return formatted

####################
### MODIFICATION ###
####################

def rename_row(df, column ,rename_map):
    # rename multiple columns
    if not isinstance(rename_map, dict) or not rename_map:
        return df.copy()
    if column not in df.columns:
        logger.save_log(f"rename_row: column '{column}' not found")
        return df.copy()

    out = df.copy()
    s_before = out[column].astype(object)
    s_after = s_before.replace(rename_map)
    out[column] = s_after

    # stats
    changed_mask = (s_before != s_after) & ~(s_before.isna() & s_after.isna())
    n_changed = int(changed_mask.sum())

    st.session_state._SV_rename_label = True

    # Per-rule hit counts (only log rules that matched)
    for old, new in rename_map.items():
        cnt = int(((s_before == old) & (s_after == new)).sum())
        if cnt > 0:
            line = f"Renamed {n_changed} label(s): '{old}' -> '{new}'"
            logger.save_log(line)
            st.session_state._LV_RenamedLabels.append(line)

    return out

##############
### RUNNER ###
##############

def labelValidaorTotal(df, label_col):
    # missing labels
    missing_labels = df[label_col].isnull().sum() > 0

    # spelling inconsistencies
    unique_labels = df[label_col].dropna().unique()
    lower_map = {}
    for label in unique_labels:
        lower = str(label).lower()
        lower_map.setdefault(lower, []).append(label)
    inconsistent_groups = {k: v for k, v in lower_map.items() if len(set(v)) > 1}
    
    # label entropy
    label_counts = df[label_col].value_counts()
    num_labels = label_counts.size
    label_probs = label_counts / label_counts.sum()
    label_entropy_value = entropy(label_probs, base=2)
    max_entropy = np.log2(num_labels) if num_labels > 0 else 0
    ratio = label_entropy_value / max_entropy if max_entropy > 0 else 0
    # interpret entropy
    if ratio == 1:
        entropy_interpretation = "Perfectly balanced label distribution."
    elif ratio >= 0.75:
        entropy_interpretation = "Labels are mostly balanced."
    elif ratio >= 0.5:
        entropy_interpretation = "Moderate label imbalance."
    elif ratio > 0:
        entropy_interpretation = "High label imbalance detected."
    else:
        entropy_interpretation = "Only one label present (no diversity)."

    # class distribution
    class_counts = df[label_col].value_counts()
    labels = class_counts.index.tolist()
    values = class_counts.values.tolist()       

    return {
        "missing_labels": missing_labels,
        "inconsistent_groups": inconsistent_groups,
        "label_entropy_value": label_entropy_value,
        "entropy_interpretation": entropy_interpretation,
        "class_counts": class_counts,
        "class_counts_labels": labels,
        "class_counts_values": values
    }