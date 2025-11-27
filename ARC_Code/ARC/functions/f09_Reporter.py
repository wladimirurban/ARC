import io
from math import ceil
from datetime import datetime
from tkinter import Canvas
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve, average_precision_score

# Local modules
import functions.f02_SchemaValidator as sv
import functions.f03_LabelValidator as lv
import functions.f04_Splitter as sd
import functions.f06_Trainer_Evaluator as te

# ReportLab core
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth

from xml.sax.saxutils import escape

# ReportLab platypus
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

# ReportLab graphics
from reportlab.graphics.shapes import (
    Drawing,
    Rect,
    Line,
    String,
    Group,
    PolyLine,
)
from reportlab.graphics.charts.piecharts import Pie

doc = SimpleDocTemplate(
        "schema_report.pdf",
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )

palette = [
        '#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD',
        '#8C564B','#E377C2','#7F7F7f','#BCBD22','#17BECf',
        '#AEC7E8','#FFBB78','#98DF8A','#FF9896','#C5B0D5',
        '#C49C94','#F7B6D2','#C7C7C7','#DBDB8D','#9EDAE5',
    ]


##############
### Styles ###
##############

_styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=_styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, spaceAfter=8)
H2 = ParagraphStyle('H2', parent=_styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, spaceBefore=10, spaceAfter=6)
H3 = ParagraphStyle('H3', parent=_styles['Heading3'], fontName='Helvetica-Bold', fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle('BODY', parent=_styles['BodyText'], fontName='Helvetica', fontSize=11, leading=14)
BOLD = ParagraphStyle('BOLD', parent=_styles['BodyText'], fontName='Helvetica-Bold', fontSize=11, leading=14)
SMALL = ParagraphStyle('SMALL', parent=_styles['BodyText'], fontName='Helvetica', fontSize=9, textColor=colors.grey)

#######################
### Footer callback ###
########################

def _on_page(canvas: Canvas, doc):
    '''Draw footer with left text, centered page number, and right placeholder logo.'''
    width, height = A4
    lb, rb = 30, width - 30
    y = 20
    canvas.setFont('Helvetica', 9)
    canvas.drawString(lb, y, 'A.R.C. - Automated Research for Cybersecurity')
    page_str = f"{canvas.getPageNumber()}"
    # center page number
    tw = canvas.stringWidth(page_str, 'Helvetica', 9)
    canvas.drawString((width - tw) / 2, y, page_str)
    # right text
    logo = 'LOGO'
    tw2 = canvas.stringWidth(logo, 'Helvetica', 9)
    canvas.drawString(rb - tw2, y, logo)

###############
### Helpers ###
###############

# General

def _kv_table(pairs: List[Tuple[str, str]], col1_w=6*cm, col2_w=10*cm):
    data = [[Paragraph('<b>Metric</b>', BODY), Paragraph('<b>Value</b>', BODY)]]
    for k, v in pairs:
        data.append([Paragraph(str(k), BODY), Paragraph(str(v), BODY)])

    tbl = Table(data, colWidths=[col1_w, col2_w], repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    return tbl

def _multi_column_list(
    items: Iterable[str],
    n_cols: int = 3,
    bullet: str = '',
    font_size: int = 10,
    row_leading: int = 12,
    gap: float = 0.5*cm
):
    '''
    Render a list of strings into an n-column table, top-to-bottom per column.
    '''
    items = list(items)
    n = len(items)
    if n == 0:
        return Paragraph('None', BODY)

    # compute rows per column, then fill column-major
    rows = ceil(n / n_cols)
    cols: List[List[str]] = []
    for c in range(n_cols):
        start = c * rows
        cols.append(items[start:start + rows])

    # pad columns to equal length
    max_rows = max(len(c) for c in cols)
    for c in range(n_cols):
        if len(cols[c]) < max_rows:
            cols[c] += [''] * (max_rows - len(cols[c]))

    # assemble row-wise for Table
    def fmt(x: str) -> Paragraph:
        return Paragraph(f"{bullet} {x}" if x else '', ParagraphStyle(
            'LIST', parent=BODY, fontSize=font_size, leading=row_leading))

    table_data = []
    for r in range(max_rows):
        table_data.append([fmt(cols[c][r]) for c in range(n_cols)])

    tbl = Table(table_data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
        # set space between columns by adding right padding except last col
    ]))
    return tbl

def df_to_table(df, col_widths=None):
    # Convert DataFrame to list of lists (including header row)
    data = [df.columns.tolist()] + df.reset_index().values.tolist()

    # Create table
    table = Table(data, colWidths=col_widths)

    # Styling
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F77B4')),  # header background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),                 # header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
    ])
    table.setStyle(style)

    return table

def make_pie_chart(labels, values, size=200, mode='lvp', font_name='Helvetica', font_size=11):
    if not labels or not values:
        return Drawing(size + 100, size + 50)
    
    # Sort labels + values alphabetically by label
    pairs = sorted(zip(labels, values), key=lambda x: str(x[0]))
    sorted_labels, sorted_values = zip(*pairs)

    total = sum(sorted_values)
    if mode == 'l':
        _labels = [
            f"{lab}" for lab, val in zip(sorted_labels, sorted_values)
        ]
    elif mode == 'lv':
        _labels = [
            f"{lab}: {val}" for lab, val in zip(sorted_labels, sorted_values)
        ]
    elif mode == 'lp':
        _labels = [
            f"{lab} ({(val / total * 100):.1f}%)" for lab, val in zip(sorted_labels, sorted_values)
        ]
    elif mode == 'lvp':
        _labels = [
            f"{lab}: {val} ({(val / total * 100):.1f}%)" for lab, val in zip(sorted_labels, sorted_values)
        ]

    drawing = Drawing(size + 100, size + 50)
    pie = Pie()
    pie.x = 50
    pie.y = 30
    pie.width = size
    pie.height = size
    pie.data = list(sorted_values)

    if sorted_labels:
        pie.labels = _labels
        pie.sideLabels = True

    # Apply font style to labels
    pie.simpleLabels = 0  # allows font customization
    pie._seriesCount = len(sorted_values)
    for i in range(len(sorted_values)):
        pie.slices[i].strokeWidth = 0
        pie.slices[i].strokeColor = None
        pie.slices[i].fontName = font_name
        pie.slices[i].fontSize = font_size

    # Apply custom colors
    color_list = [colors.HexColor(c) for c in palette]
    for i in range(len(sorted_values)):
        pie.slices[i].fillColor = color_list[i % len(color_list)]

    drawing.add(pie)
    drawing.hAlign = 'CENTER'
    return drawing

# Label Validator

def plot_timeline(font_name: str = 'Helvetica', font_size: int = 10, height: int = 110):
    df = st.session_state._DF
    ts_col = st.session_state._TimeStampCol
    label_col = st.session_state._LabelCol

    s = (
        df.dropna(subset=[ts_col, label_col])
          .sort_values(ts_col)[[ts_col, label_col]]
          .copy()
    )
    if s.empty:
        return None

    s[ts_col] = pd.to_datetime(s[ts_col], errors='coerce')
    s = s.dropna(subset=[ts_col])
    if s.empty:
        return None

    run_id = (s[label_col] != s[label_col].shift()).cumsum()
    segments = (
        s.groupby(run_id)
         .agg(start=(ts_col, 'min'),
              end=(ts_col, 'max'),
              label=(label_col, 'first'))
         .reset_index(drop=True)
    )
    segments['end'] = segments.apply(
        lambda r: r['end'] if r['end'] > r['start'] else r['start'] + pd.Timedelta(milliseconds=1),
        axis=1,
    )

    t_min = segments['start'].min()
    t_max = segments['end'].max()
    span = (t_max - t_min).total_seconds()
    if span <= 0:
        return None

    labels_order = sorted(df[label_col].dropna().astype(str).unique().tolist())

    color_map = {lab: colors.HexColor(palette[i % len(palette)]) for i, lab in enumerate(labels_order)}
    
    # LAYOUT
    width = 500
    gap = 6
    # Legend
    box = 8
    item_pad = 12
    legend_width = width - gap*2

    # Measure legend item widths precisely
    items = [
        (lab, box + gap + stringWidth(str(lab), font_name, font_size) + item_pad)
        for lab in labels_order
    ]

    # Pack legend items into rows
    rows = []
    cur_row = []
    cur_w = 0.0
    for lab, iw in items:
        if cur_row and cur_w + iw > legend_width:
            rows.append(cur_row)
            cur_row = [(lab, iw)]
            cur_w   = iw
        else:
            cur_row.append((lab, iw))
            cur_w += iw
    if cur_row:
        rows.append(cur_row)
    
    legend_height = (len(rows) * (font_size + gap))+gap

    # Label
    tick_lenght = 6
    n_ticks = 5
    label_height = font_size*2+gap*3 + tick_lenght

    # Plot
    label_width = stringWidth(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), font_name, font_size)
    left_margin = gap + label_width/2
    right_margin = width-left_margin
    plot_width = width-left_margin*2
    block_heights = 24
    tick_lenght = 6
    
    block_heights = height - legend_height - label_height

    graph_y = height - legend_height - block_heights

    d = Drawing(width, height)
    
    # Draw main line
    d.add(Line(left_margin, graph_y, right_margin, graph_y, strokeColor=colors.black, strokeWidth=1))
    
    # Draw blocks
    def tx(t): return left_margin + plot_width * ((t - t_min).total_seconds() / span)
    min_px_w = 0.5
    for _, row in segments.iterrows():
        x0 = tx(row['start'])
        x1 = tx(row['end'])
        w  = max(x1 - x0, min_px_w)
        y  = graph_y #- block_heights/2.0
        col = color_map.get(str(row['label']), colors.lightgrey)
        d.add(Rect(x0, y, w, block_heights, fillColor=col, strokeColor=None, strokeWidth=0))
    
    # Draw ticks and x labels
    for i in range(n_ticks):
        frac = i/(n_ticks-1) if n_ticks > 1 else 0
        xt = left_margin + plot_width*frac

        # Tick marks
        d.add(
            Line(
                xt, graph_y, 
                xt, graph_y - tick_lenght,
                strokeColor=colors.black, strokeWidth=1
            )
        )

        t_val = t_min + pd.to_timedelta(span*frac, unit='s')
        txt = t_val.strftime('%Y-%m-%d %H:%M')

        # x labels
        g = Group()
        g.add(
            String(
                0, 0, 
                txt, fontName=font_name, fontSize=font_size,
                fillColor=colors.black, textAnchor='middle'
            )
        )
        g.translate(xt, graph_y - tick_lenght - gap - font_size)
        d.add(g)
        
    # Draw legend
    legend_top_y = height - font_size - gap
    for r_idx, row in enumerate(rows):
        row_w = sum(iw for _, iw in row)
        start_x = (width - row_w) / 2.0
        y = legend_top_y - r_idx * (box + gap)
        x = start_x
        for lab, iw in row:
            col = color_map.get(str(lab), colors.lightgrey)
            d.add(Rect(x, y, box, box, fillColor=col, strokeColor=None, strokeWidth=0))
            d.add(String(x + box + gap, y, str(lab),
                         fontName=font_name, fontSize=font_size,
                         fillColor=colors.black, textAnchor='start'))
            x += iw

    # Drax x label
    d.add(String(
        width / 2,
        6,
        'Time',
        fontName=font_name,
        fontSize=font_size,
        fillColor=colors.black,
        textAnchor='middle'
    ))
    
    return d

def plot_timebin(
    data: pd.DataFrame,
    *,
    font_name: str = 'Helvetica',
    font_size: int = 10,
    y_ticks: int = 4,
    y_axis_title: str = 'Count',
):
    # ---- validate & reshape ----
    if data is None or data.empty or 'interval' not in data.columns:
        return None

    count_cols = [c for c in data.columns if c.endswith(' count')]
    if not count_cols:
        return None

    labels: List[str] = [c[:-6].strip() for c in count_cols]
    labels_order: List[str] = sorted(labels)

    counts_only = data[['interval'] + count_cols].copy()
    counts_only = counts_only.rename(columns={f"{lab} count": lab for lab in labels})
    # melt to long form (interval, label, Count)
    df_plot = counts_only.melt(id_vars='interval', var_name='label', value_name='Count')
    # ensure categorical order for labels
    df_plot['label'] = pd.Categorical(df_plot['label'], categories=labels_order, ordered=True)

    # per-interval stacked heights & max for scaling
    interval_order = data['interval'].tolist()
    totals_by_interval = df_plot.groupby('interval')['Count'].sum().reindex(interval_order).fillna(0)
    max_total = max(1, int(totals_by_interval.max()))

    # ---- colors ----
    colors_hex = {lab: palette[i % len(palette)] for i, lab in enumerate(labels_order)}
    color_map = {lab: colors.HexColor(hexc) for lab, hexc in colors_hex.items()}

    # ---- layout ----

    width = 500
    gap = 6

    ## Legend
    box = 8
    item_pad = 12
    legend_width = width - gap*2

    # Label
    label_width = stringWidth(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), font_name, font_size)
    label_height = label_width + font_size+gap*2

    # Plot
    tick_lenght = 6
    number_width = stringWidth(f"{max_total}", font_name, font_size)
    x_left = gap*2 + font_size + number_width + tick_lenght
    x_right = width - gap
    plot_y = label_height
    plot_width = width - x_left - gap
    p_h = 100
    
    plot_height = p_h + gap

    items = [
        (lab, box + gap + stringWidth(str(lab), font_name, font_size) + item_pad) for lab in labels_order
    ]

    # Pack items into rows
    rows = []
    cur_row = []
    cur_w = 0.0
    for lab, iw in items:
        if cur_row and cur_w + iw > legend_width:
            rows.append(cur_row)
            cur_row = [(lab, iw)]
            cur_w   = iw
        else:
            cur_row.append((lab, iw))
            cur_w += iw
    if cur_row:
        rows.append(cur_row)
    
    legend_height = (len(rows) * (font_size + gap))+gap

    total_height = legend_height + plot_height + label_height

    d = Drawing(width, total_height)
    d.add(Rect(
        0, 0,
        width, total_height,
        strokeColor=colors.black,
        fillColor=None,
        strokeWidth=1
    ))

    # Plot
    # y-axis line
    d.add(Line(x_left, plot_y, x_left, plot_y + p_h, strokeColor=colors.black, strokeWidth=1))
    # x-axis line
    d.add(Line(x_left, plot_y, x_right, plot_y, strokeColor=colors.black, strokeWidth=1))
    
    # y ticks & labels
    for i in range(y_ticks + 1):
        frac = i / y_ticks
        y = plot_y + frac * p_h
        # tick
        d.add(Line(x_left, y, x_left - tick_lenght, y, strokeColor=colors.black, strokeWidth=1))
        # label
        val = int(round(frac * max_total))
        d.add(String(x_left - 8, y - font_size/3, f"{val}", textAnchor='end',
                     fontName=font_name, fontSize=font_size, fillColor=colors.black))
    
    # bars
    n_bins = len(interval_order)
    bar_w = (plot_width - (n_bins - 1) * gap) / n_bins

    # for each interval, draw stacked rectangles bottom-up
    for idx, interval_label in enumerate(interval_order):
        interval_df = df_plot[df_plot['interval'] == interval_label].sort_values('label')
        base_y = plot_y
        x_l = x_left + idx * (bar_w + gap)
        # stack by labels in labels_order
        for lab in labels_order:
            v = float(interval_df.loc[interval_df['label'] == lab, 'Count'].sum())
            if v <= 0:
                continue
            h = (v / max_total) * p_h
            d.add(
                Rect(
                    x_l, base_y, 
                    bar_w, h,
                    fillColor=color_map.get(lab, colors.lightgrey),
                    strokeColor=None, strokeWidth=0))
            base_y += h

    # x tick labels (interval strings)
    for idx, interval_label in enumerate(interval_order):
        x_center = x_left + idx * (bar_w + gap) + bar_w / 2
        txt = str(interval_label)
        parts = txt.split(' - ', 1)

        # --- tick line ---
        d.add(Line(
            x_center, plot_y, 
            x_center, plot_y - tick_lenght,
            strokeColor=colors.black,
            strokeWidth=0.8
        ))

        # --- rotated multiline label ---
        g = Group()
        for line_idx, part in enumerate(parts):
            s = String(
                0, -line_idx * (font_size),  # vertical offset per line
                part,
                fontName=font_name,
                fontSize=font_size,
                fillColor=colors.black,
                textAnchor='end',
            )
            g.add(s)

        g.translate(x_center, plot_y - tick_lenght - 2)  # place just under the tick
        g.rotate(60)
        d.add(g)
        
    # axis titles
    g = Group()
    s = String(0, 0, 'Count',
               fontName=font_name,
               fontSize=font_size,
               fillColor=colors.black,
               textAnchor='middle')
    g.add(s)
    g.translate(gap + font_size, plot_y + plot_height / 2.0)
    g.rotate(90)

    d.add(g)
    
    d.add(String(
        width / 2,
        6,
        'Time',
        fontName=font_name,
        fontSize=font_size,
        fillColor=colors.black,
        textAnchor='middle'
    ))

    # Draw rows centered within full width
    legend_top_y = total_height - font_size - 6
    for r_idx, row in enumerate(rows):
        row_w = sum(iw for _, iw in row)
        start_x = (width - row_w) / 2.0          # CENTER over full drawing width
        y = legend_top_y - r_idx * (box + gap)
        x = start_x
        for lab, iw in row:
            col = color_map.get(str(lab), colors.lightgrey)
            d.add(Rect(x, y, box, box, fillColor=col, strokeColor=None, strokeWidth=0))
            d.add(String(x + box + gap, y, str(lab),
                         fontName=font_name, fontSize=font_size,
                         fillColor=colors.black, textAnchor='start'))
            x += iw

    return d

def time_bin_table(out: pd.DataFrame, page_width=A4[0], font_name='Helvetica', font_size=10, margin=60):
    if out is None or out.empty or 'interval' not in out.columns:
        return None

    labels = sorted(set(c.rsplit(' ', 1)[0] for c in out.columns if c != 'interval'))

    # header as Paragraphs (rotation works better on flowables than plain strings)
    styles = getSampleStyleSheet()
    hstyle = styles['BodyText'].clone('hdr')
    hstyle.fontName = font_name
    hstyle.fontSize = font_size
    hstyle.leading  = font_size

    header = [Paragraph('Interval', hstyle)] + [Paragraph(lab, hstyle) for lab in labels]

    data = [header]
    for _, row in out.iterrows():
        interval_str = str(row['interval'])
        if ' - ' in interval_str:
            a, b = interval_str.split(' - ', 1)
            interval_str = f"{a}\n{b}"
        row_data = [interval_str]
        for lab in labels:
            pct_col, count_col = f"{lab} %", f"{lab} count"
            val = (
                f"{row[count_col]}\n({row[pct_col]:.1f}%)"
                if pct_col in out.columns and count_col in out.columns
                else '-'
            )
            row_data.append(val)
        data.append(row_data)

    total_w = 500#page_width - 2*margin
    w0 = total_w * 0.2
    w_rest = (total_w - w0) / max(len(labels), 1)
    col_widths = [w0] + [w_rest]*len(labels)

    tbl = Table(data, colWidths=col_widths, hAlign='CENTER')

    st = TableStyle([
        ('FONTNAME', (0, 1), (-1, -1), font_name),
        ('FONTSIZE', (0, 1), (-1, -1), font_size),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        # rotate label headers (columns 1..N) by 90°
        ('ROTATE', (1, 0), (-1, 0), 90),
        # adjust paddings so rotated text sits nicely
        ('LEFTPADDING', (1, 0), (-1, 0), 2),
        ('RIGHTPADDING', (1, 0), (-1, 0), 2),
        ('BOTTOMPADDING', (1, 0), (-1, 0), 6),
        ('ALIGN', (1, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (1, 0), (-1, 0), 'MIDDLE'),
    ])
    tbl.setStyle(st)
    return tbl

def series_to_table(series, col_widths=None):
    '''
    Convert a Pandas Series (like value_counts) to a 2-column ReportLab table.
    '''
    # Build data: header row + rows from series
    data = [['Label', 'Count']] + [[str(idx), int(val)] for idx, val in series.items()]

    # Create table
    table = Table(data, colWidths=col_widths)

    # Apply styling
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F77B4')),  # header bg
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),                 # header text
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
    ])
    table.setStyle(style)

    return table

# Split

def split_stacked_bar(
    dist_df: pd.DataFrame,
    font_name: str = 'Helvetica',
    font_size: int = 10,
):

    stack_cols = ['train_%', 'val_%', 'test_%']
    for c in stack_cols:
        if c not in dist_df.columns:
            raise ValueError(f"Missing required column '{c}' in dist_df")

    # data keys (for lookup) vs display strings (for x labels)
    label_keys = dist_df.index.tolist()
    label_disp = [str(x) for x in label_keys]
    n_bins = len(label_keys)

    # colors and legend display names (order: Train, Validate, Test)
    col_order = ['train_%', 'val_%', 'test_%']
    legend_names = {'train_%': 'Train', 'val_%': 'Validate', 'test_%': 'Test'}
    colors_map = {
        'train_%': colors.HexColor('#1F77B4'),
        'val_%':   colors.HexColor('#FF7F0E'),
        'test_%':  colors.HexColor('#2CA02C'),
    }

    totals = dist_df[stack_cols].sum(axis=1)
    max_total = float(totals.max()) if float(totals.max()) > 0 else 1.0

# ---- layout ----
    width = 500
    gap = 6

    ## Legend
    box = 8
    item_pad = 12
    legend_height = gap*2 + font_size

    # Label
    label_width = max(stringWidth(str(lbl), font_name, font_size) for lbl in label_keys)
    label_height = label_width + gap

    # Plot
    tick_lenght = 6
    number_width = stringWidth(f"{max_total}", font_name, font_size)
    x_left = gap + number_width + tick_lenght
    x_right = width - gap
    plot_y = label_height
    plot_width = width - x_left - gap
    plot_h = 100
    
    plot_height = plot_h + gap

    total_height = plot_height + label_height + legend_height
    d = Drawing(width, total_height)
    d.add(Rect(
        0, 0,
        width, total_height,
        strokeColor=colors.black,
        fillColor=None,
        strokeWidth=1
    ))

    # axes
    d.add(Line(x_left, plot_y, x_left, plot_y + plot_h, strokeColor=colors.black, strokeWidth=1))
    d.add(Line(x_left, plot_y, x_right, plot_y, strokeColor=colors.black, strokeWidth=1))

    # y ticks
    y_ticks = 5
    for i in range(y_ticks + 1):
        frac = i / y_ticks
        y = plot_y + frac * plot_h
        val = frac * max_total
        d.add(Line(x_left - tick_lenght, y, x_left, y, strokeColor=colors.black, strokeWidth=1))
        d.add(String(x_left - tick_lenght - gap, y - font_size/3, f"{val:.0f}",
                     fontName=font_name, fontSize=font_size, textAnchor='end'))
    

    # bars
    #gap = 10.0
    total_gap = (n_bins - 1) * gap
    usable_w = max(1.0, plot_width - total_gap)
    bar_w = max(3.0, usable_w / n_bins)

    for idx, key in enumerate(label_keys):
        x_l = x_left + idx * (bar_w + gap)
        base_y = plot_y

        row_total = float(dist_df.loc[key, stack_cols].sum())
        if row_total <= 0:
            row_total = 1.0

        for col in col_order:
            v = float(dist_df.loc[key, col])
            if v <= 0:
                continue
            h = (v / max_total) * plot_h

            d.add(Rect(x_l, base_y, bar_w, h,
                       fillColor=colors_map[col], strokeColor=None, strokeWidth=0))

            if h >= font_size:
                pct_val = (v / row_total) * 100.0
                pct_text = f"{pct_val:.0f}%" if abs(pct_val - round(pct_val)) < 1e-9 else f"{pct_val:.1f}%"
                cx = x_l + bar_w / 2.0
                cy = (base_y + h / 2.0)-font_size/3

                fill = colors_map[col]
                r, g, b = fill.red, fill.green, fill.blue
                luminance = 0.299*r + 0.587*g + 0.114*b
                text_col = colors.white if luminance < 0.5 else colors.black

                d.add(String(cx, cy, pct_text,
                             fontName=font_name, fontSize=font_size,
                             fillColor=text_col, textAnchor='middle'))

            base_y += h

        # rotated x label (70°)
        gx = Group()
        sx = String(0, 0, label_disp[idx], fontName=font_name, fontSize=font_size,
                    fillColor=colors.black, textAnchor='end')
        gx.add(sx)
        gx.translate(x_l + bar_w/2.0, plot_y - 6)
        gx.rotate(70)
        d.add(gx)

    # legend centered at top (labels: Train, Validate, Test)
    items = col_order
    legend_w = sum(stringWidth(legend_names[it], font_name, font_size)
                   + box + gap + item_pad for it in items)
    start_x = (width - legend_w) / 2.0
    y_leg = total_height - gap - font_size
    x = start_x
    for it in items:
        name = legend_names[it]
        d.add(Rect(x, y_leg - 1, box, box, fillColor=colors_map[it], strokeColor=None))
        d.add(String(x + box + gap, y_leg, name,
                     fontName=font_name, fontSize=font_size,
                     fillColor=colors.black, textAnchor='start'))
        x += stringWidth(name, font_name, font_size) + box + gap + item_pad

    return d

def rl_distribution_table_styled(
    dist_df,
    page_width=A4[0],
    font_name='Helvetica',
    font_size=10,
    margin=60
):
    '''
    Build a ReportLab table:
    Columns: Label | Train | Validate | Test
    Each split column shows 'count' on first line and '(percent%)' on second line.
    Styled to match rl_time_bin_table, but without rotated headers.
    '''
    if dist_df is None or dist_df.empty:
        return None

    required = {'train_count','val_count','test_count','train_%','val_%','test_%'}
    missing = required - set(dist_df.columns)
    if missing:
        raise ValueError(f"dist_df missing columns: {sorted(missing)}")

    # Styles
    styles = getSampleStyleSheet()
    hstyle = styles['BodyText'].clone('hdr')
    hstyle.fontName = font_name
    hstyle.fontSize = font_size
    hstyle.leading  = font_size

    # Header
    header = [
        Paragraph('Label', hstyle),
        Paragraph('Train', hstyle),
        Paragraph('Validate', hstyle),
        Paragraph('Test', hstyle),
    ]

    data = [header]

    # Rows
    for lab in dist_df.index.tolist():
        tr_c = int(dist_df.at[lab, 'train_count'])
        va_c = int(dist_df.at[lab, 'val_count'])
        te_c = int(dist_df.at[lab, 'test_count'])

        tr_p = float(dist_df.at[lab, 'train_%'])
        va_p = float(dist_df.at[lab, 'val_%'])
        te_p = float(dist_df.at[lab, 'test_%'])

        row = [
            str(lab),
            f"{tr_c} ({tr_p:.1f}%)",
            f"{va_c} ({va_p:.1f}%)",
            f"{te_c} ({te_p:.1f}%)",
        ]
        data.append(row)

    # Column widths
    total_w = 500
    w0 = total_w * 0.35       # Label column wider
    w_rest = (total_w - w0) / 3.0
    col_widths = [w0, w_rest, w_rest, w_rest]

    tbl = Table(data, colWidths=col_widths, hAlign='CENTER')

    st = TableStyle([
        ('FONTNAME', (0, 1), (-1, -1), font_name),
        ('FONTSIZE', (0, 1), (-1, -1), font_size),

        ('ALIGN', (0, 1), (0, -1), 'LEFT'),     # Label col
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),  # Numeric cols
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),

        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),

        # Header paddings
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
    ])

    # Zebra stripes for body
    for r in range(1, len(data)):
        if r % 2 == 0:
            st.add('BACKGROUND', (0, r), (-1, r), colors.whitesmoke)

    tbl.setStyle(st)
    return tbl

# Evaluation

def generalEvaluation(
        model_P, 
        model_NP,
        compare: bool = False, 
        font_size: int = 10, 
        font_name: str = 'Helvetica'
    ):
    gap = 6
    width = 500

    if compare == True:
        height = gap*5 + font_size*4.5
    else:
        height = gap*3 + font_size*2.5
    
    tt_P = model_P['train_time_sec']
    ms_P = model_P['model_size_bytes']
    nf_P = model_P['n_features']
    
    d = Drawing(width, height)
    d.add(String(width/6, height-gap-font_size, 'Train time (s)', fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
    d.add(String(width/2, height-gap-font_size, 'Model size', fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
    d.add(String(width*5/6, height-gap-font_size, 'Number of features', fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
    if compare == True:
        tt_NP = model_NP['train_time_sec']
        ms_NP = model_NP['model_size_bytes']
        nf_NP = model_NP['n_features']

        tt_delta = te._pct_delta(tt_P, tt_NP)
        ms_delta = te._pct_delta(ms_P, ms_NP)
        nf_delta = (nf_P - nf_NP) / nf_NP * 100

        for x in [1,5,9]:
            d.add(String(width*x/12, height-gap*2-font_size*2, 'RAW', fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
        for x in [3,7,11]:
            d.add(String(width*x/12, height-gap*2-font_size*2, 'PRE', fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))

        d.add(String(width*3/12, height-gap*3-font_size*3.5, f"{tt_P:.3f}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width*7/12, height-gap*3-font_size*3.5, f"{te.fmt_bytes(ms_P)}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width*11/12, height-gap*3-font_size*3.5, f"{nf_P}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width*1/12, height-gap*3-font_size*3.5, f"{tt_NP:.3f}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width*5/12, height-gap*3-font_size*3.5, f"{te.fmt_bytes(ms_NP)}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width*9/12, height-gap*3-font_size*3.5, f"{nf_NP}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))

        d.add(String(width*3/12, height-gap*4-font_size*4.5, f"{tt_delta:+.2f}%", fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width*7/12, height-gap*4-font_size*4.5, f"{ms_delta:+.2f}%", fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width*11/12, height-gap*4-font_size*4.5, f"{nf_delta:+.2f}%", fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))

    else:
        d.add(String(width/6, height-gap*2-font_size*2.5, f"{tt_P:.3f}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width/2, height-gap*2-font_size*2.5, f"{te.fmt_bytes(ms_P)}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))
        d.add(String(width*5/6, height-gap*2-font_size*2.5, f"{nf_P}", fontName=font_name, fontSize=font_size*1.5, fillColor=colors.black, textAnchor='middle'))
    return d

def metricsEvaluation(
        model_P: Dict = [], 
        model_NP: Dict = [],
        compare: bool = False, 
        font_size: int = 10, 
        font_name: str = 'Helvetica'
    ):
    gap = 6
    if compare == True:
        header_heigt = gap*4 + font_size*2
    else:
        header_heigt = gap*2 + font_size

    value_height = gap*12 + font_size*6
    height = value_height + header_heigt
    width = 500
    l_width = stringWidth('Interference time [s]', font_name, font_size) + gap*2
    v_width = width - l_width

    d = Drawing(width, height)
    d.add(Rect(
        0, 0,
        width, height,
        strokeColor=colors.black,
        fillColor=None,
        strokeWidth=1
    ))

    #Label column
    temp_y = gap
    for l in ['Interference time [s]', 'ROC-AUC', 'Recall (macro)', 'Percision (macro)', 'F1 (macro)', 'Accurancy']:
        d.add(String(gap, temp_y, l, fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='start'))
        temp_y += gap*2 + font_size

    #Header
    x_temp = l_width + v_width/6
    d.add(String(l_width/2, height - gap - font_size, 'Metric', fontName=font_name, fontSize=font_size, textAnchor='middle'))
    for l in ['Train', 'Validate', 'Test']:
        d.add(String(x_temp, height - gap - font_size, l, fontName=font_name, fontSize=font_size, textAnchor='middle'))
        x_temp += v_width/3

    # Compare
    if compare == True:
        # Background color
        x_temp = l_width
        for c in [colors.HexColor('#E0E0E0'), colors.HexColor('#CCCCCC'), colors.HexColor('#B3B3B3')]:
            for x in range(0,3):
                d.add(Rect(
                    x_temp+v_width*x/3, 0,
                    v_width*1/9, value_height+gap*2+font_size,
                    strokeColor=None,
                    fillColor=c,
                    strokeWidth=1
                ))
            x_temp += v_width*1/9
        # Small vertical lines
        for x in [1,2,4,5,7,8]:
            d.add(Line(l_width + v_width*x/9, 0, l_width + v_width*x/9, value_height+gap*2+font_size, strokeColor=colors.black, strokeWidth=1))
        d.add(Line(0, value_height+gap*2+font_size, width, value_height+gap*2+font_size, strokeColor=colors.black, strokeWidth=1))
        x_temp = l_width + v_width/18
        for t in ['RAW', 'PRE', 'Δ']:
            for x in range(0,3):
                d.add(String(x_temp+v_width*x/3, value_height+gap, t, fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
            x_temp += v_width/9

        x_temp = l_width + v_width/18
        for split in ['train', 'validation', 'test']:
            p = model_P.get(split)
            np = model_NP.get(split)
            m_p = p['metrics']
            m_np = np['metrics']
            temp_y = gap
            for v in [
                f"{np['inference_time_sec']:.4f}", 
                f"{m_np['roc_auc']*100:.2f}%", 
                f"{m_np['recall_macro']*100:.2f}%", 
                f"{m_np['precision_macro']*100:.2f}%", 
                f"{m_np['f1_macro']*100:.2f}%", 
                f"{m_np['accuracy']*100:.2f}%"
                ]:
                d.add(String(x_temp, temp_y, v, fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
                temp_y += gap*2 + font_size
            
            temp_y = gap
            for v in [
                f"{p['inference_time_sec']:.4f}", 
                f"{m_p['roc_auc']*100:.2f}%", 
                f"{m_p['recall_macro']*100:.2f}%", 
                f"{m_p['precision_macro']*100:.2f}%", 
                f"{m_p['f1_macro']*100:.2f}%", 
                f"{m_p['accuracy']*100:.2f}%"
                ]:
                d.add(String(x_temp + v_width/9, temp_y, v, fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
                temp_y += gap*2 + font_size
            
            temp_y = gap
            for v in [
                f"{p['inference_time_sec']-np['inference_time_sec']:.4f}", 
                f"{m_p['roc_auc']*100 - m_np['roc_auc']*100:.2f} pp", 
                f"{m_p['recall_macro']*100 - m_np['recall_macro']*100:.2f}pp", 
                f"{m_p['precision_macro']*100 - m_np['precision_macro']*100:.2f}pp", 
                f"{m_p['f1_macro']*100 - m_np['f1_macro']*100:.2f}pp", 
                f"{m_p['f1_macro']*100 - m_np['accuracy']*100:.2f}pp"
                ]:
                d.add(String(x_temp + v_width*2/9, temp_y, v, fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
                temp_y += gap*2 + font_size
            
            x_temp += v_width/3
    else:
        x_temp = l_width + v_width/6
        for split in ['train', 'validation', 'test']:
            p = model_P.get(split)
            m = p['metrics']
            temp_y = gap
            for v in [f"{p['inference_time_sec']:.4f}", f"{m['roc_auc']*100:.2f}%", f"{m['recall_macro']*100:.2f}%", f"{m['precision_macro']*100:.2f}%", f"{m['f1_macro']*100:.2f}%", f"{m['accuracy']*100:.2f}%"]:
                d.add(String(x_temp, temp_y, v, fontName=font_name, fontSize=font_size, fillColor=colors.black, textAnchor='middle'))
                temp_y += gap*2 + font_size
            x_temp += v_width/3
    # Vertical lines
    for x in [1,2]:
        d.add(Line(l_width + v_width*x/3, 0, l_width + v_width*x/3, height, strokeColor=colors.black, strokeWidth=1))
    #Horizontal lines
    for y in range(1,7):
        d.add(Line(0, value_height*y/6, width, value_height*y/6, strokeColor=colors.black, strokeWidth=1))
    d.add(Line(l_width, 0, l_width, height, strokeColor=colors.black, strokeWidth=1))
    
    return d

def rl_confusion_matrix(
    cm: np.ndarray,
    class_names=None,
    *,
    width: int = 500,
    height: int = 500,
    font_name: str = 'Helvetica',
    font_size: int = 9,
    show_colorbar: bool = True,
):
    cm = np.asarray(cm)
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1], 'cm must be square'
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    else:
        class_names = [str(x) for x in class_names]
        assert len(class_names) == n, 'class_names length must match cm'

    # --- layout ---
    gap = 6

    label_w   = max(stringWidth(s, font_name, font_size) for s in class_names)*0.85
    
    cbar_w        = 14 if show_colorbar else 0
    cbar_gap      = 8  if show_colorbar else 0

    top_margin    = gap
    bot_margin = 2*gap + font_size + label_w
    right_margin  = gap + cbar_w + cbar_gap
    left_margin = 2*gap + font_size + label_w

    p_w = width - left_margin - right_margin
    p_h = height - top_margin - bot_margin

    if p_w < p_h:
        plot_h = p_w
        plot_w = p_w
    else:
        plot_h = p_h
        plot_w = p_h

    cell_w = plot_w / n
    cell_h = plot_h / n

    x0 = (width-plot_w)/2
    y0 = bot_margin

    d = Drawing(width, height)

    # --- color mapping (white → red) ---
    col0 = colors.white
    col1 = colors.HexColor('#FF4B4B')

    def lerp(a, b, t):
        return a + (b - a) * t

    def lerp_color(c0, c1, t):
        return colors.Color(
            lerp(c0.red,   c1.red,   t),
            lerp(c0.green, c1.green, t),
            lerp(c0.blue,  c1.blue,  t),
        )

    vmax = float(cm.max()) if cm.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    # --- draw cells ---
    for i in range(n):
        for j in range(n):
            val = float(cm[i, j])
            t = min(1.0, max(0.0, val / vmax))
            fill = lerp_color(col0, col1, t)
            x = x0 + j * cell_w
            y = y0 + (n - 1 - i) * cell_h

            d.add(Rect(x, y, cell_w, cell_h, fillColor=fill, strokeColor=colors.black, strokeWidth=0.3))

            # text (count) centered; black is visible on light background
            cx = x + cell_w/2
            cy = y + cell_h/2
            d.add(String(cx, cy, f"{int(val)}", fontName=font_name, fontSize=font_size,
                         fillColor=colors.black, textAnchor='middle'))

        # --- axis ticks ---
    for j, name in enumerate(class_names):
        x = x0 + j * cell_w + cell_w/2
        g = Group()
        s = String(0, 0, name,
                   fontName=font_name, fontSize=font_size,
                   fillColor=colors.black, textAnchor='end')
        g.add(s)
        g.translate(x+3, y0 - gap)   # just below the axis
        g.rotate(45)             # rotate 45°
        d.add(g)

    for i, name in enumerate(class_names):
        y = y0 + (n - 1 - i) * cell_h + cell_h/2
        g = Group()
        s = String(0, 0, name,
                   fontName=font_name, fontSize=font_size,
                   fillColor=colors.black, textAnchor='end')
        g.add(s)
        g.translate(x0 - gap, y)   # just left of the axis
        g.rotate(45)             # rotate 45°
        d.add(g)

    # --- axis titles ---
    d.add(String(x0 + plot_w/2, gap, 'Predicted',
                 fontName=font_name, fontSize=font_size+1,
                 fillColor=colors.black, textAnchor='middle'))
    gy = Group()
    gy.add(String(0, 0, 'Actual', fontName=font_name, fontSize=font_size+1,
                  fillColor=colors.black, textAnchor='middle'))
    gy.translate(x0 - label_w, y0 + plot_h/2)
    gy.rotate(90)
    d.add(gy)

    # --- colorbar ---
    if show_colorbar:
        steps = 50
        cb_x = x0 + plot_w + cbar_gap
        cb_y = y0
        cb_h = plot_h
        for k in range(steps):
            t0 = k / steps
            yk = cb_y + t0 * cb_h
            fill = lerp_color(col0, col1, t0)
            d.add(Rect(cb_x, yk, cbar_w, cb_h/steps + 1, fillColor=fill, strokeColor=None))
        d.add(Rect(cb_x, cb_y, cbar_w, cb_h, fillColor=None, strokeColor=colors.black, strokeWidth=0.6))
        for frac, label in [(0.0, '0'), (0.5, f"{int(round(vmax/2))}"), (1.0, f"{int(vmax)}")]:
            yy = cb_y + frac * cb_h
            d.add(Line(cb_x + cbar_w, yy, cb_x + cbar_w + 4, yy, strokeColor=colors.black, strokeWidth=0.6))
            d.add(String(cb_x + cbar_w + 6, yy - 3, label, fontName=font_name, fontSize=font_size,
                         fillColor=colors.black, textAnchor='start'))

    return d

def plot_roc_legend(
    model,
    X,
    y,
    class_names=None,
    *,
    font_name: str = 'Helvetica',
    font_size: int =  8,
):
    width = 500
    height = 0
    # --- get probabilities ---
    classes = getattr(model, 'classes_', None)
    classes = np.array(classes)

    # class names
    if class_names is not None:
        if len(class_names) != len(classes):
            raise ValueError('Length of class_names must match number of classes in model')
        legend_names = list(class_names)
    else:
        legend_names = [str(c) for c in classes]

    # --- compute ROC curves ---
    label_collection = []
    for k, cls in enumerate(classes):
        name = legend_names[k] if k < len(legend_names) else f"class {cls}"
        label_collection.append({'name': name})

    # colors for curves
    stroke_colors = [colors.HexColor(palette[i % len(palette)]) for i in range(len(label_collection))]

    # Build legend items (text + color)
    legend_items = []
    for idx, curve in enumerate(label_collection):
        name = curve['name']
        legend_items.append((name, stroke_colors[idx]))

    # --- pack legend rows to fit available width ---
    box = 8
    gap = 6
    item_pad = 12

    max_row_width = width - gap*2  # use almost full drawing width
    rows = []
    cur_row = []
    cur_w = 0.0
    for name, col in legend_items:
        iw = box + gap + stringWidth(name, font_name, font_size) + item_pad
        if cur_row and (cur_w + iw) > max_row_width:
            rows.append(cur_row)
            cur_row = [(name, col, iw)]
            cur_w = iw
        else:
            cur_row.append((name, col, iw))
            cur_w += iw
    if cur_row:
        rows.append(cur_row)

    # total legend height
    gap = 6
    row_h = gap + font_size
    legend_height = len(rows) * row_h + gap
    # --- layout with legend at top ---
    d = Drawing(width, legend_height)
    # Legend
    if rows:
        top_y = legend_height - gap
        y = top_y
        for r_idx, row in enumerate(rows):
            row_w = sum(item_w for _, _, item_w in row) - (item_pad if row else 0)
            start_x = (width - row_w) / 2.0
            x = start_x
            for name, col, item_w in row:
                # box
                d.add(Rect(x, y - box + 1, box, box,
                           fillColor=col, strokeColor=None))
                # text
                d.add(String(x + box + gap, y - box + 1, name,
                             fontName=font_name, fontSize=font_size,
                             fillColor=colors.black, textAnchor='start'))
                x += item_w
            y -= (row_h)
    return d

def rl_plot_roc(
    model,
    X,
    y,
    class_names=None,
    *,
    width: int = 500,
    height: int = 420,
    font_name: str = 'Helvetica',
    font_size: int =  10,
    grid_steps: int = 5,
    show_auc: bool = True,
    show_legend: bool = True
):
    from sklearn.metrics import roc_curve, auc

    try:
        probas = model.predict_proba(X)
    except Exception:
        return Drawing(width, height)

    classes = getattr(model, 'classes_', None)
    if probas is None or classes is None:
        return Drawing(width, height)

    classes = np.array(classes)

    # class names
    if class_names is not None:
        if len(class_names) != len(classes):
            raise ValueError('Length of class_names must match number of classes in model')
        legend_names = list(class_names)
    else:
        legend_names = [str(c) for c in classes]

    # --- compute ROC curves ---
    roc_curves = []
    if probas.ndim == 1 or (probas.ndim == 2 and probas.shape[1] == 2):
        p1 = probas if probas.ndim == 1 else probas[:, 1]
        fpr, tpr, _ = roc_curve(y, p1)
        roc_auc = auc(fpr, tpr)
        name = legend_names[1] if len(legend_names) == 2 else (legend_names[-1] if legend_names else 'pos')
        roc_curves.append({'name': name, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})
    else:
        for k, cls in enumerate(classes):
            y_bin = (np.asarray(y) == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, probas[:, k])
            roc_auc = auc(fpr, tpr)
            name = legend_names[k] if k < len(legend_names) else f"class {cls}"
            roc_curves.append({'name': name, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc})

    # colors for curves
    stroke_colors = [colors.HexColor(palette[i % len(palette)]) for i in range(len(roc_curves))]

    # Build legend items (text + color)
    legend_items = []
    for idx, curve in enumerate(roc_curves):
        name = curve['name']
        if show_auc:
            name = f"{name} (AUC={curve['auc']:.3f})"
        legend_items.append((name, stroke_colors[idx]))

    # --- pack legend rows to fit available width ---
    box = 8
    gap = 6
    item_pad = 16

    max_row_width = width - gap*2  # use almost full drawing width
    rows = []
    cur_row = []
    cur_w = 0.0
    for name, col in legend_items:
        iw = box + gap + stringWidth(name, font_name, font_size) + item_pad
        if cur_row and (cur_w + iw) > max_row_width:
            rows.append(cur_row)
            cur_row = [(name, col, iw)]
            cur_w = iw
        else:
            cur_row.append((name, col, iw))
            cur_w += iw
    if cur_row:
        rows.append(cur_row)

    # total legend height
    gap = 6
    row_h = gap + font_size

    if show_legend == True:
        legend_height = len(rows) * row_h + gap
    else:
        legend_height = gap*2

    label_height = gap*2 + font_size*2
    plot_h = height - legend_height - label_height

    # --- layout with legend at top ---
    d = Drawing(width, height)

    left_margin = gap*3 + font_size + stringWidth('1.0', font_name, font_size)
    right_margin = gap + stringWidth('1.0', font_name, font_size)/2
    bottom_margin =  gap*3 + font_size*2

    plot_w = width  - left_margin - right_margin
    #plot_h = height

    x0 = left_margin
    y0 = bottom_margin

    # Legend
    if show_legend == True:
        if rows:
            top_y = height - gap
            y = top_y
            for r_idx, row in enumerate(rows):
                row_w = sum(item_w for _, _, item_w in row) - (item_pad if row else 0)
                start_x = (width - row_w) / 2.0
                x = start_x
                for name, col, item_w in row:
                    # box
                    d.add(Rect(x, y - box + 1, box, box,
                            fillColor=col, strokeColor=None))
                    # text
                    d.add(String(x + box + gap, y - box + 1, name,
                                fontName=font_name, fontSize=font_size,
                                fillColor=colors.black, textAnchor='start'))
                    x += item_w
                y -= (row_h)

    # axes
    d.add(Line(x0, y0, x0 + plot_w, y0, strokeColor=colors.black, strokeWidth=0.8))         # x-axis
    d.add(Line(x0, y0, x0, y0 + plot_h, strokeColor=colors.black, strokeWidth=0.8))         # y-axis

    # grid + ticks (0..1)
    steps = max(2, int(grid_steps))
    for i in range(steps + 1):
        frac = i / steps
        gx = x0 + frac * plot_w
        gy = y0 + frac * plot_h
        d.add(Line(gx, y0, gx, y0 + plot_h, strokeColor=colors.HexColor('#E5E7EB'), strokeWidth=0.4))
        d.add(Line(x0, gy, x0 + plot_w, gy, strokeColor=colors.HexColor('#E5E7EB'), strokeWidth=0.4))
        d.add(String(gx, y0 - gap - font_size, f"{frac:.1f}", fontName=font_name, fontSize=font_size,
                     fillColor=colors.black, textAnchor='middle'))
        d.add(String(x0 - gap, gy - 3, f"{frac:.1f}", fontName=font_name, fontSize=font_size,
                     fillColor=colors.black, textAnchor='end'))

    # chance diagonal
    d.add(Line(x0, y0, x0 + plot_w, y0 + plot_h, strokeColor=colors.HexColor("#9CA3Af"),
               strokeWidth=0.8, strokeDashArray=[3,3]))

    # plot curves
    for idx, curve in enumerate(roc_curves):
        fpr = np.clip(curve['fpr'], 0, 1)
        tpr = np.clip(curve['tpr'], 0, 1)
        pts = [(x0 + float(fx) * plot_w, y0 + float(ty) * plot_h) for fx, ty in zip(fpr, tpr)]
        if len(pts) >= 2:
            d.add(PolyLine(pts, strokeColor=stroke_colors[idx], strokeWidth=1.8))

    # axis titles
    d.add(String(x0 + plot_w/2, gap, 'FPR', fontName=font_name, fontSize=font_size,
                 fillColor=colors.black, textAnchor='middle'))
    gy = Group()
    gy.add(String(0, 0, 'TPR', fontName=font_name, fontSize=font_size,
                  fillColor=colors.black, textAnchor='middle'))
    gy.translate(gap + font_size, y0 + plot_h/2)
    gy.rotate(90)
    d.add(gy)

    return d

def rl_plot_pr(
    model,
    X,
    y,
    class_names=None,
    *,
    width: int = 500,
    height: int = 420,
    font_name: str = 'Helvetica',
    font_size: int = 10,
    grid_steps: int = 5,
    show_legend: bool = True
):
    try:
        probas = model.predict_proba(X)
    except Exception:
        return Drawing(width, height)

    classes = getattr(model, 'classes_', None)
    if probas is None or classes is None:
        return Drawing(width, height)

    classes = np.asarray(classes)
    if class_names is None:
        legend_names = [str(c) for c in classes]
    else:
        if len(class_names) != len(classes):
            legend_names = [str(c) for c in classes]
        else:
            legend_names = list(class_names)

    pr_curves = []  # each item: {'name', 'recall', 'precision', 'ap'}

    # Binary (2 probs) or single prob
    if probas.ndim == 1 or (probas.ndim == 2 and probas.shape[1] == 2):
        p1 = probas if probas.ndim == 1 else probas[:, 1]
        precision, recall, _ = precision_recall_curve(y, p1)
        ap = average_precision_score(y, p1)
        name = legend_names[1] if len(legend_names) > 1 else 'pos'
        pr_curves.append({'name': f"{name} (AP={ap:.3f})",
                          'recall': recall, 'precision': precision, 'ap': ap})
    else:
        # Multiclass one-vs-rest
        y_arr = np.asarray(y)
        for k, cls in enumerate(classes):
            y_bin = (y_arr == cls).astype(int)
            precision, recall, _ = precision_recall_curve(y_bin, probas[:, k])
            ap = average_precision_score(y_bin, probas[:, k])
            name = legend_names[k] if k < len(legend_names) else f"class {cls}"
            pr_curves.append({'name': f"{name} (AP={ap:.3f})",
                              'recall': recall, 'precision': precision, 'ap': ap})

    # --- colors (Plotly-like palette) ---
    stroke_colors = [colors.HexColor(palette[i % len(palette)]) for i in range(len(pr_curves))]

    # Legend rows packed at the very top, centered across full width
    box = 8
    gap = 6
    item_pad = 16

    items = [(curve['name'], stroke_colors[i]) for i, curve in enumerate(pr_curves)]
    max_row_width = width * 0.94

    rows = []
    cur_row, cur_w = [], 0.0
    for name, col in items:
        iw = box + gap + stringWidth(name, font_name, font_size) + item_pad
        if cur_row and (cur_w + iw) > max_row_width:
            rows.append(cur_row)
            cur_row = [(name, col, iw)]
            cur_w = iw
        else:
            cur_row.append((name, col, iw))
            cur_w += iw
    if cur_row:
        rows.append(cur_row)

    row_h = font_size + gap

    if show_legend == True:
        legend_height = len(rows) * row_h + gap
    else: 
        legend_height = gap*2

    # Plot area after legend
    left_margin = gap*3 + font_size + stringWidth('0.0', font_name, font_size)
    right_margin= gap*2
    bottom_margin = gap*3 + font_size*2
    plot_w = width  - left_margin - right_margin

    plot_h = height - legend_height - bottom_margin
    x0, y0 = left_margin, bottom_margin

    # --- layout ---
    d = Drawing(width, height)
    d.add(Rect(
            0, 0,
            width, height,
            strokeColor=colors.black,
            fillColor=None,
            strokeWidth=1
        ))
    if show_legend == True:
    # Draw the legend (top, centered, wrapped)
        if rows:
            top_y = height - gap
            y = top_y
            for row in rows:
                row_w = sum(iw for _, _, iw in row) - (item_pad if row else 0)
                start_x = (width - row_w) / 2.0
                x = start_x
                for name, col, iw in row:
                    d.add(Rect(x, y - box + 1, box, box, fillColor=col, strokeColor=None))
                    d.add(String(x + box + gap, y - box + 1, name,
                                fontName=font_name, fontSize=font_size,
                                fillColor=colors.black, textAnchor='start'))
                    x += iw
                y -= (row_h)

    # Axes
    d.add(Line(x0, y0, x0 + plot_w, y0, strokeColor=colors.black, strokeWidth=0.8))         # x-axis (Recall)
    d.add(Line(x0, y0, x0, y0 + plot_h, strokeColor=colors.black, strokeWidth=0.8))         # y-axis (Precision)

    # Grid + ticks (0..1)
    steps = max(2, int(grid_steps))
    for i in range(steps + 1):
        frac = i / steps
        gx = x0 + frac * plot_w
        gy = y0 + frac * plot_h
        # grid lines
        d.add(Line(gx, y0, gx, y0 + plot_h, strokeColor=colors.HexColor('#E5E7EB'), strokeWidth=0.4))
        d.add(Line(x0, gy, x0 + plot_w, gy, strokeColor=colors.HexColor('#E5E7EB'), strokeWidth=0.4))
        # tick labels
        d.add(String(gx, y0 - 10, f"{frac:.1f}", fontName=font_name, fontSize=font_size,
                     fillColor=colors.black, textAnchor='middle'))
        d.add(String(x0 - 10, gy - 3, f"{frac:.1f}", fontName=font_name, fontSize=font_size,
                     fillColor=colors.black, textAnchor='end'))

    # Plot PR curves
    for i, curve in enumerate(pr_curves):
        # Clamp to [0,1]
        r = np.clip(curve['recall'],    0, 1)
        p = np.clip(curve['precision'], 0, 1)
        pts = [(x0 + float(rx) * plot_w, y0 + float(py) * plot_h) for rx, py in zip(r, p)]
        if len(pts) >= 2:
            d.add(PolyLine(pts, strokeColor=stroke_colors[i], strokeWidth=1.8))

    # Axis titles
    d.add(String(x0 + plot_w/2, y0 - 26, 'Recall',
                 fontName=font_name, fontSize=font_size+1, fillColor=colors.black, textAnchor='middle'))
    gy = Group()
    gy.add(String(0, 0, 'Precision',
                  fontName=font_name, fontSize=font_size+1, fillColor=colors.black, textAnchor='middle'))
    gy.translate(x0 - 34, y0 + plot_h/2)
    gy.rotate(90)
    d.add(gy)

    

    return d

def rl_feature_importance(
        model, 
        feature_names=None, 
        top_k=25,
        width=500, 
        height=600,
        font_name='Helvetica', 
        font_size=9,
        bar_color='#FF4B4B',
        y_label='Feature',
        x_label='Importance',
        frame=False,              # set True to draw an outer border
    ):
    # --- extract importances ---
    clf = model
    if hasattr(model, 'named_steps') and "clf" in model.named_steps:
        clf = model.named_steps['clf']

    importances = getattr(clf, 'feature_importances_', None)
    if importances is None:
        return None

    n_features = len(importances)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    # sort + select top_k
    idx = np.argsort(importances)[::-1][:top_k]
    imp_sorted = np.asarray(importances)[idx]
    names_sorted = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]

    if len(imp_sorted) == 0:
        return Drawing(width, height)

    # --- measure label width to drive left margin ---
    gap = 6
    max_label_w = max(stringWidth(str(name), font_name, font_size) for name in names_sorted)
    label_pad = 10           # space between label text and y-axis
    y_title_gap = 14         # extra gap to place rotated y-title

    # margins
    left_margin   = max_label_w + gap*3 + font_size
    right_margin  = gap*2 + stringWidth('00.000', font_name, font_size)
    top_margin    = 0
    bottom_margin = gap*2 + font_size

    # drawing
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, strokeColor=colors.black, fillColor=None, strokeWidth=1))

    plot_w = width - left_margin - right_margin
    plot_h = height - bottom_margin
    x0, y0 = left_margin, bottom_margin

    # axes
    d.add(Line(x0, y0, x0, y0 + plot_h, strokeColor=colors.black, strokeWidth=0.8))
    d.add(Line(x0, y0, x0 + plot_w, y0, strokeColor=colors.black, strokeWidth=0.8))

    # scaling
    max_val = float(imp_sorted.max()) if float(imp_sorted.max()) > 0 else 1.0
    n = len(imp_sorted)
    row_h = plot_h / n
    bar_h = row_h * 0.7
    gap   = row_h * 0.3

    # draw bars (bottom up so largest importance is at top, like Plotly)
    fill_col = colors.HexColor(bar_color)
    for i, (name, val) in enumerate(zip(names_sorted[::-1], imp_sorted[::-1])):
        y = y0 + i * row_h + (row_h - bar_h) / 2.0
        w = (val / max_val) * plot_w
        # bar
        d.add(Rect(x0, y, w, bar_h, fillColor=fill_col, strokeColor=None))
        # label on left (right-aligned)
        d.add(String(x0 - gap, y + bar_h/2-3, str(name),
                     fontName=font_name, fontSize=font_size,
                     textAnchor='end', fillColor=colors.black))
        # value on bar end
        d.add(String(x0 + w + gap, y + bar_h/2-3, f"{val:.3f}",
                     fontName=font_name, fontSize=font_size,
                     textAnchor='start', fillColor=colors.black))

    # x-axis title (centered below axis)
    d.add(String(x0 + plot_w/2, gap, x_label,
                 fontName=font_name, fontSize=font_size+1,
                 textAnchor='middle', fillColor=colors.black))

    # y-axis title rotated 90° and placed left of the longest label
    gy = Group()
    gy.add(String(0, 0, y_label,
                  fontName=font_name, fontSize=font_size+1,
                  textAnchor='middle', fillColor=colors.black))
    # place it at horizontal x = (left edge of labels) - extra gap
    y_title_x = left_margin - (max_label_w + label_pad + y_title_gap/2)
    y_title_y = y0 + plot_h / 2.0
    gy.translate(y_title_x, y_title_y)
    gy.rotate(90)
    d.add(gy)

    return d

#############
### Parts ###
#############

def dataLoadReport_story(story: List):
    story.append(Paragraph('Datas loader', H2))

    if not st.session_state.get('_DL_DataLoaded', False):
        story.append(Paragraph('No dataset loaded.', BODY))
        return

    # Dataset name
    ds_name = st.session_state.get('_DL_Filename', '(unknown)')
    story.append(Paragraph(f"Dataset name: {ds_name}", BODY))

    # Multiple files (if applicable)
    if st.session_state.get('_DL_Mode') == 'multiple':
        story.append(Paragraph('Dataset was built with multiple files:', BODY))
        files = st.session_state.get('_DL_UploadedFiles', [])
        story.append(_multi_column_list([f for f in files], n_cols=1))

    # Timestamp + Label
    if st.session_state.get('_HasTimeStamp', False):
        ts_text = f"Timestamp column: {st.session_state.get('_TimeStampCol','(unknown)')}"
    else:
        ts_text = 'No timestamp column detected.'

    if st.session_state.get('_HasLabel', False):
        label_text = f"Label column: {st.session_state.get('_LabelCol','(unknown)')}"
    else:
        label_text = 'No label column detected.'
    story.append(Paragraph(f"{ts_text}", BODY))
    story.append(Paragraph(f"{label_text}", BODY))

    # Columns in dataset (4 columns display)
    story.append(Paragraph('Columns in dataset:', BODY))
    df = st.session_state.get("_DF", None)
    if isinstance(df, pd.DataFrame):
        cols = sorted(df.columns.tolist())
        story.append(_multi_column_list(cols, n_cols=4, bullet=''))
    else:
        story.append(Paragraph('N/A', BODY))
    story.append(Spacer(1, 12))

    # Footer note
    story.append(Paragraph(' ', SMALL))  # small spacer line to keep layout consistent

def schemaValidatorReport_story(story: List):
    df = st.session_state.get("_DF", None)
    hastimestamp = st.session_state.get('_HasTimeStamp', False)
    tiemsatmpcol = st.session_state.get('_TimeStampCol', None)

    story.append(Paragraph('Schema validator', H1))
    story.append(Paragraph('Insights', H2))

    if not st.session_state.get('_DL_DataLoaded', False):
        story.append(Paragraph('No dataset loaded.', BODY))
        return

    sv_data = sv.schemaValidatorTotal(df, hastimestamp, tiemsatmpcol)

    # General
    story.append(Paragraph('General', H3))
    story.append(Paragraph(f"Total records: {sv_data['total_records']:,}", BODY))
    dups = sv_data['duplicates']
    if dups == 0:
        story.append(Paragraph('Duplicate records: None', BODY))
    else:
        dup_perc = (dups / sv_data['total_records']) * 100
        story.append(Paragraph(f"Duplicate records: {dups} ({dup_perc:.2f}% of total)", BODY))
    story.append(Spacer(1, 8))

    # Sparsity
    story.append(Paragraph('Sparsity', H3))
    tot_miss = sv_data['sparsity']['Total missing percent']
    story.append(Paragraph(f"Percentage of missing values: {tot_miss}%", BODY))
    col_miss = sv_data['sparsity']['Col missing']  # pandas Series
    story.append(Paragraph(f"{len(col_miss)} columns have missing values:", BODY))
    items = [f"{c}: {round(float(p), 2)}%" for c, p in col_miss.items()]
    story.append(_multi_column_list(items, n_cols=3, bullet='-'))
    story.append(Spacer(1, 8))

    # Granularity
    story.append(Paragraph('Granularity', H3))
    story.append(Paragraph(f"Detected Granularity: {sv_data['granularity']}", BODY))
    story.append(Spacer(1, 8))

    # Features
    story.append(Paragraph('Features', H3))
    story.append(Paragraph(f"Total features: {sv_data['features']['Total Features']}", BODY))
    if sv_data['features_unique'] == True:
        story.append(Paragraph('All features are unique', BODY))
    else:
        story.append(Paragraph('Duplicated features detected', BODY))
    mixed_cols = sv_data.get('mixed_type_cols', [])
    if not mixed_cols:
        story.append(Paragraph('No columns with mixed data types detected.', BODY))
    else:
        story.append(Paragraph(f"{len(mixed_cols)} columns with mixed data types detected:", BODY))
        story.append(_multi_column_list(mixed_cols, n_cols=3))
    
    values = [
        len(sv_data['features']['Numerical']), 
        len(sv_data['features']['Categorical']), 
        len(sv_data['features']['Bool']), 
        len(sv_data['features']['Object']), 
        len(sv_data['features']['Datetime'])
        ]
    labels = ['Numerical', 'Categorical', 'Bool', 'Object', 'Datetime']

    story.append(make_pie_chart(labels, values, size=100))

    block = []
    
    if len(sv_data['features']['Numerical']) == 0:
        block.append(Paragraph('Numeric features: None', BOLD))
    else:
        block.append(Paragraph(f"Numeric features ({len(sv_data['features']['Numerical'])}):", BOLD))
        block.append(_multi_column_list(sv_data['features']['Numerical'], n_cols=4))
    
    if len(sv_data['features']['Categorical']) == 0:
        block.append(Paragraph('Categorical features: None', BOLD))
    else:
        block.append(Paragraph(f"Categorical features ({len(sv_data['features']['Categorical'])}):", BOLD))
        block.append(_multi_column_list(sv_data['features']['Categorical'], n_cols=4))

    if len(sv_data['features']['Bool']) == 0:
        block.append(Paragraph('Bool features: None', BOLD))
    else:
        block.append(Paragraph(f"Bool features ({len(sv_data['features']['Bool'])}):", BOLD))
        block.append(_multi_column_list(sv_data['features']['Bool'], n_cols=4))
    
    if len(sv_data['features']['Object']) == 0:
        block.append(Paragraph('Object features: None', BOLD))
    else:
        block.append(Paragraph(f"Object features ({len(sv_data['features']['Object'])}):", BOLD))
        block.append(_multi_column_list(sv_data['features']['Object'], n_cols=4))
        
    if len(sv_data['features']['Datetime']) == 0:
        block.append(Paragraph('Datetime features: None', BOLD))
    else:
        block.append(Paragraph(f"Datetime features ({len(sv_data['features']['Datetime'])}):", BOLD))
        block.append(_multi_column_list(sv_data['features']['Datetime'], n_cols=4))
    
    story.append(KeepTogether(block))
    
    block = []
    block.append(Paragraph('Time Column Analysis', H3))
    if st.session_state._HasTimeStamp == False:
        story.append(Paragraph('No timestamp columns avalible', BODY))
    else:
        time_dup = sv_data.get('time_dup', None)
        if sv_data['time_dup'] == True:
            block.append(Paragraph('Duplicate timestamps found.', BODY))
        else:
            block.append(Paragraph('No duplicate timestamps found.', BODY))
        
        if sv_data['time_sorted'] == True:
            block.append(Paragraph(f"Dataset is sorted by timestamp column ({st.session_state._TimeStampCol}).", BODY))
        else:
            block.append(Paragraph(f"Dataset is not sorted by timestamp column ({st.session_state._TimeStampCol}).", BODY))
        
        block.append(Paragraph(f"Start: {sv_data['time_analysis']['Start']}", BODY))
        block.append(Paragraph(f"End: {sv_data['time_analysis']['End']}", BODY))
        block.append(Paragraph(f"Duration: {sv_data['time_analysis']['Duration']}", BODY))
        block.append(Paragraph(f"Most common interval: {sv_data['time_analysis']['Most common interval']}", BODY))
        block.append(Paragraph(f"Average interval: {sv_data['time_analysis']['Average interval']}", BODY))
    story.append(KeepTogether(block))

    block = []
    block.append(Paragraph('Modifications', H2))
    if st.session_state._SV_NormCol:
        block.append(Paragraph('Columns normalized', BODY))
    if st.session_state._SV_RenameCol:
        block.append(Paragraph('Columns renamed', BODY))
    if st.session_state._SV_DropCol:
        block.append(Paragraph('Columns dropped', BODY))
    if st.session_state._SV_SortByT:
        block.append(Paragraph('Dataset sorted by timestamp column', BODY))
        if st.session_state._SV_DeltaT:
            block.append(Paragraph('Delta time column added', BODY))
    if st.session_state._SV_DropDup:
        block.append(Paragraph('Duplicate records dropped', BODY))
    story.append(KeepTogether(block))

def labelValidatorReport_story(story: List):
    story.append(Paragraph('Label validator', H1))
    if not st.session_state._DL_DataLoaded:
        story.append(Paragraph('No dataset loaded.', BODY))
        story.append(Spacer(1, 12))
        return
    if st.session_state._HasLabel == False:
        story.append(Paragraph('No label column detected.', BODY))
        story.append(Spacer(1, 12))
        return
    
    df = st.session_state.get("_DF", None)
    labelcol = st.session_state.get('_LabelCol', None)
    hastimestamp = st.session_state.get('_HasTimeStamp', False)
    timestampcol = st.session_state.get('_TimeStampCol', None)
    tdtimeline = st.session_state.get('_LV_TDTimeline', False)
    tdtime = st.session_state.get('_LV_TDTime', False)
    tdrecords = st.session_state.get('_LV_TDRecords', False)
    tdnumbins = st.session_state.get('_LV_TDNumBins', 10)

    lv_data = lv.labelValidaorTotal(df, labelcol)

    block = []
    block.append(Paragraph('Labeling consitency', H2))
    # Missing labels
    if lv_data['missing_labels'] == True:
        block.append(Paragraph('Missing labels detected.', BODY))
    else:
        block.append(Paragraph('No missing labels detected.', BODY))

    # Inconsistent labeling
    if lv_data['inconsistent_groups']:
        block.append(Paragraph('Possible label spelling inconsistencies:', BODY))
        block.append(_multi_column_list(lv_data['inconsistent_groups'], n_cols=2))
    else:
        block.append(Paragraph('The label spelling seems to be consistent', BODY))
    story.append(KeepTogether(block))
    
    # Class distribution
    block = []
    block.append(Paragraph('Class distribution', H2))

    # Class counts
    block.append(Paragraph('Class counts', H3))
    class_counts = lv_data['class_counts']
    labels = class_counts.index.tolist()
    values = class_counts.values.tolist()
    block.append(make_pie_chart(labels, values, size=100))
    
    # Rare classes
    block.append(Paragraph('Rare classes', H3))
    rare_classes = lv.get_rare_classes(df, labelcol, st.session_state._LV_InputRareClasses)
    if rare_classes.empty:
        block.append(Paragraph(f"No class represents less then {st.session_state._LV_InputRareClasses}%", BODY))
    else:
        formatted = ', '.join(
            f"{label}: {ratio:.2%}" for label, ratio in rare_classes.items()
        )
        block.append(Paragraph(f"Rare classes (>{st.session_state._LV_InputRareClasses}%) detected:", BODY))
        block.append(_multi_column_list(formatted, 2))
    
    # Dominant classes
    block.append(Paragraph('Dominant classes', H3))
    dominant_classes = lv.get_dominant_classes(df, labelcol, st.session_state._LV_InputDominantClasses)
    if dominant_classes.empty:
        block.append(Paragraph(f"No class represents more then {st.session_state._LV_InputDominantClasses}%", BODY))
    else:
        formatted = ', '.join(
            f"{label}: {ratio:.2%}" for label, ratio in dominant_classes.items()
        )
        block.append(Paragraph(f"Dominant classes (>{st.session_state._LV_InputDominantClasses}%) detected:", BODY))
        block.append(_multi_column_list(formatted, 2))

    story.append(KeepTogether(block))

    #Temporal drift
    
    story.append(Paragraph('Temporal drift', H2))
    if hastimestamp == True:
        if tdtimeline == True:
            block = []
            block.append(Paragraph('Label distribution timelinee', H3))
            block.append(plot_timeline())
            story.append(KeepTogether(block))

        if tdtime == True:
            block = []
            block.append(Paragraph('Label distribution over time-bins grouped by time', H3))
            if tdnumbins > 10:
                table = lv.get_timebin(df, labelcol, timestampcol, 'time', 10)
            else:
                table = lv.get_timebin('time', tdnumbins)
            block.append(plot_timebin(table))
            block.append(Spacer(1,6))
            tbl = time_bin_table(table)
            block.append(tbl)
            story.append(KeepTogether(block))
            
        if tdrecords == True:
            block = []
            block.append(Paragraph('Label distribution over time-bins grouped by records', H3))
            if tdnumbins > 10:
                table = lv.get_timebin(df, labelcol, timestampcol, 'records', 10)
            else:
                table = lv.get_timebin('records', tdnumbins)
            block.append(plot_timebin(table))
            block.append(Spacer(1,6))
            tbl = time_bin_table(table)
            block.append(tbl)
            story.append(KeepTogether(block))
        
    else:
        block.append(Paragraph('No timestamo detected', BODY))

    block = []
    block.append(Paragraph('Modifications', H2))
    if st.session_state._LV_RenameLabel:
        block.append(Paragraph('Label column renamed', BODY))
    else:
        block.append(Paragraph('No modifications applied', BODY))
        
def splitReport_story(story: List):
    block = []
    block.append(Paragraph('Splitter', H1))
    if not st.session_state._DL_DataLoaded:
        block.append(Paragraph('No dataset loaded.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._HasLabel == False:
        block.append(Paragraph('No label column detected.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._SP_IsSplit == False:
        block.append(Paragraph('Dataset was mot splitted.', BODY))
        story.append(KeepTogether(block))
        return
    
    splitmethod = st.session_state._SP_SplitMethod
    testsize = st.session_state._SP_TestSize
    valsize = st.session_state._SP_ValSize

    X_train = st.session_state._SP_X_Train
    y_train = st.session_state._SP_y_Train
    X_val = st.session_state._SP_X_Validate
    y_val = st.session_state._SP_y_Validate
    X_test = st.session_state._SP_X_Test
    y_test = st.session_state._SP_y_Test

    
    block.append(Paragraph('Split', H2))
    block.append(Paragraph(f"Split method: {splitmethod}", BODY))
    block.append(Paragraph(f"Train set: {len(X_train)} samples ({100 - testsize - valsize}%)", BODY))
    block.append(Paragraph(f"Validate set: {len(X_val)} samples ({valsize}%)", BODY))
    block.append(Paragraph(f"Test set: {len(X_test)} samples ({testsize}%)", BODY))
    story.append(KeepTogether(block))

    block = []
    block.append(Paragraph('Quality checks', H2))
    block.append(Paragraph('Unseen labels', H3))
    unseen = sd.getunseen(y_train, y_val, y_test)
    if unseen['unseen_val']:
        block.append(Paragraph('Unseen labels in the validation set (not present in training):', BODY))
        block.append(_multi_column_list(unseen['unseen_val'], 3))
    else:
        block.append(Paragraph('No unseen labels in the validation set.', BODY))
    
    if unseen['unseen_test']:
        block.append(Paragraph('Unseen labels in the test set (not present in training):', BODY))
        block.append(_multi_column_list(unseen['unseen_test'], 3))
    else:
        block.append(Paragraph('No unseen labels in the test set.', BODY))
    story.append(KeepTogether(block))
    
    story.append(Paragraph('Label distribution', H3))
    dist_df = sd.getDistributionLabel(y_train, y_val, y_test)
    story.append(split_stacked_bar(dist_df = dist_df))
    story.append(Spacer(1,6))
    story.append(rl_distribution_table_styled(dist_df = dist_df))

def evaluationReport_story(story: List):
    block = []
    block.append(Paragraph('Evaluator', H1))
    if not st.session_state._DL_DataLoaded:
        block.append(Paragraph('No dataset loaded.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._HasLabel == False:
        block.append(Paragraph('No label column detected.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._SP_IsSplit == False:
        block.append(Paragraph('Dataset was mot splitted.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._TE_PTrained == False:
        block.append(Paragraph('Model not trained.', BODY))
        story.append(KeepTogether(block))
        return
    
    
    block.append(Paragraph(f"Training method: {st.session_state._TE_Model}", BODY))
    story.append(KeepTogether(block))

    results = st.session_state._TE_Model
    results2 = st.session_state._MODEL_P

    block = []
    block.append(Paragraph('General', H2))
    block.append(generalEvaluation(model_P=results, model_NP=results, compare=False))
    story.append(KeepTogether(block))
    
    block = []
    block.append(Paragraph('Metrics', H2))
    block.append(metricsEvaluation(model_P=results, model_NP=results, compare=False))
    story.append(KeepTogether(block))

    block = []
    block.append(Paragraph('Confusion Matrix', H2))
    class_names = [name for name, _id in sorted(st.session_state._PP_LE.items(), key=lambda x: x[1])]
    block.append(Paragraph('Train', H3))
    cm_train = np.array(results['train']['metrics']['confusion_matrix'])
    block.append(rl_confusion_matrix(cm_train, class_names, width=500, height=300))
    story.append(KeepTogether(block))

    block = []
    block.append(Paragraph('Validation', H3))
    cm_validate = np.array(results['validation']['metrics']['confusion_matrix'])
    block.append(rl_confusion_matrix(cm_validate, class_names, width=500, height=300))
    story.append(KeepTogether(block))

    block = []
    block.append(Paragraph('Test', H3))
    cm_test = np.array(results['test']['metrics']['confusion_matrix'])
    block.append(rl_confusion_matrix(cm_test, class_names, width=500, height=300))
    story.append(KeepTogether(block))

    X_train = st.session_state._PP_X_train
    y_train = st.session_state._PP_y_train
    X_val = st.session_state._PP_X_validate
    y_val = st.session_state._PP_y_validate
    X_test = st.session_state._PP_X_test
    y_test = st.session_state._PP_y_test
    block = []
    block.append(Paragraph('ROC', H2))
    block.append(plot_roc_legend(results['model'], X_train, y_train, class_names=class_names))
    block.append(Paragraph('Train', H3))
    block.append(rl_plot_roc(results['model'], X_train, y_train, class_names=class_names, width=500, height=150, show_legend=False))
    block.append(Paragraph('Validation', H3))
    block.append(rl_plot_roc(results['model'], X_val, y_val, class_names=class_names, width=500, height=150, show_legend=False))
    block.append(Paragraph('Test', H3))
    block.append(rl_plot_roc(results['model'], X_test, y_test, class_names=class_names, width=500, height=150, show_legend=False))
    story.append(KeepTogether(block))

    block = []
    block.append(Paragraph('PR', H2))
    block.append(plot_roc_legend(results['model'], X_train, y_train, class_names=class_names))
    block.append(Paragraph('Train', H3))
    block.append(rl_plot_pr(results['model'], X_train, y_train, class_names=class_names, width=500, height=150, show_legend=False))
    block.append(Paragraph('Validation', H3))
    block.append(rl_plot_pr(results['model'], X_val, y_val, class_names=class_names, width=500, height=150, show_legend=False))
    block.append(Paragraph('Test', H3))
    block.append(rl_plot_pr(results['model'], X_test, y_test, class_names=class_names, width=500, height=150, show_legend=False))
    story.append(KeepTogether(block))

    feat_names = getattr(X_train, 'columns', None)
    story.append(rl_feature_importance(results['model'], feature_names=feat_names, top_k=25))

def compareReport_story(story: List):
    block = []
    block.append(Paragraph('Comparer', H1))
    if not st.session_state._DL_DataLoaded:
        block.append(Paragraph('No dataset loaded.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._HasLabel == False:
        block.append(Paragraph('No label column detected.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._SP_IsSplit == False:
        block.append(Paragraph('Dataset was mot splitted.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._TE_PTrained == False:
        block.append(Paragraph('Model not trained.', BODY))
        story.append(KeepTogether(block))
        return
    if st.session_state._C_NPTrained == False:
        block.append(Paragraph('RAW model not trained.', BODY))
        story.append(KeepTogether(block))
        return
    
    
    block.append(Paragraph(f"Training method: {st.session_state._TE_Model}", BODY))
    story.append(KeepTogether(block))

    results = st.session_state._MODEL_P
    results2 = st.session_state._MODEL_NP

    block = []
    block.append(Paragraph('General', H2))
    block.append(generalEvaluation(model_P=results, model_NP=results, compare=True))
    story.append(KeepTogether(block))
    
    block = []
    block.append(Paragraph('Metrics', H2))
    block.append(metricsEvaluation(model_P=results, model_NP=results2, compare=True))
    story.append(KeepTogether(block))

def logReport_story(story: List):

    story.append(Paragraph('Logger', H1))
    story.append(Spacer(1, 6))

    log = st.session_state._LogData
    if not log:
        story.append(Paragraph('No log entries.', BODY))
        return

    tail = log[::-1]

    for entry in tail:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            ts, msg = entry
            ts_html = f"<b>{escape(str(ts))}</b>"
            msg_html = escape(str(msg)).replace('\n', '<br/>')
            block = [
                Paragraph(ts_html, BODY),
                Paragraph(msg_html, BODY),
                Spacer(1, 6),
            ]
            story.append(KeepTogether(block))
        else:
            story.append(Paragraph(escape(str(entry)), BODY))
            story.append(Spacer(1, 6))

###################
### Entry point ###
###################

def create_pdf_story(): #filename: str
    buf = io.BytesIO() ###
    doc = SimpleDocTemplate(
        buf, #filename,
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )

    story: List = []
    # Cover/title-like header (optional)
    #ds_name = st.session_state.get('_DataLoader_FileName', '(unknown dataset)')
    #story.append(Paragraph(f"Dataset: {ds_name}', BODY))
    #story.append(Paragraph(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), SMALL))
    #story.append(Spacer(1, 12))

    if st.session_state._R_DL:
        dataLoadReport_story(story)
        story.append(PageBreak())
    
    if st.session_state._R_SV:
        schemaValidatorReport_story(story)
        story.append(PageBreak())
    
    if st.session_state._R_LV:
        labelValidatorReport_story(story)
        story.append(PageBreak())

    if st.session_state._R_S:
        splitReport_story(story)
        story.append(PageBreak())

    if st.session_state._R_TE:
        evaluationReport_story(story)
        story.append(PageBreak())
    
    if st.session_state._R_C:
        compareReport_story(story)
        story.append(PageBreak())
    
    if st.session_state._R_L:
        logReport_story(story)
        story.append(PageBreak())

    # Build
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buf.seek(0)
    return buf.getvalue()
