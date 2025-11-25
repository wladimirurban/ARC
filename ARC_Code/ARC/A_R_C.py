import streamlit as st

import functions.f00_VarInitialisation as vi
import functions.f00_Sidebar as sidebar

#import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"

if "_First_Start" not in st.session_state:
        st.session_state._First_Start = True

if st.session_state._First_Start == True:
    st.session_state._First = False
    vi.init()

st.set_page_config(page_title="A.R.C.", layout="wide")
st.title("A.R.C. - Automated Research for Cybersecurity")

sidebar.sidebar()

st.markdown(
        """

Welcome! This Streamlit app is a **modular pipeline for preparing and assessing cybersecurity datasets**. It helps you load raw data, validate structure and labels, split properly (including time-aware splits), apply transparent preprocessing, train baseline ML models, and **generate reproducible reports**—all in one place.

---

## What you can do here

- **Ingest data** from CSV/TXT/JSON or parse PCAP/PCAPNG into flows/sessions.
- **Validate** schema (types, duplicates, timestamp integrity) and **labels** (consistency, class balance, drift).
- **Split** data with **Random**, **Stratified**, or **Time-based** strategies—plus quality checks to avoid leakage.
- **Preprocess** via explicit, toggleable steps (cleaning, encoding, selection, scaling, imbalance handling, time features).
- **Train & evaluate** baseline models (RF, Gradient Boosting/CatBoost, SVM, MLP, Logistic Regression).
- **Compare** “minimal raw” vs. “fully preprocessed” pipelines to show the true impact of preparation.
- **Export** everything you need for reproducibility (CSV, JSON, PKL models, and a consolidated **PDF report**).

---

## Who is this for?

- Researchers and engineers who need **trustworthy**, **auditable**, and **comparable** ML experiments.
- Anyone who has wrestled with **inconsistent features**, **noisy labels**, **class imbalance**, or **time leakage** in public datasets.

---

## The A.R.C. flow at a glance

1. **Data Loader**  
   Upload CSV/TXT/JSON or PCAP/PCAPNG. Configure parsing/aggregation (e.g., flows), set **label** and **timestamp** columns, and preview the dataset.

2. **Schema Validator**  
   Detect duplicates, mixed dtypes, missing values, feature duplicates/correlation, and **temporal integrity** (ranges, gaps, monotonicity). Apply safe fixes (rename columns, sort by time, drop duplicates) with full logging.

3. **Label Validator**  
   Check **class distribution & entropy**, rare/dominant classes, spelling inconsistencies, and **temporal drift** of labels. Decide on merging/keeping rare classes before training.

4. **Splitter**  
   Create train/val/test with **Random**, **Stratified**, or **Time-based** splits. Inspect per-split class balance and unseen labels. Seeds and cut-offs are recorded for reproducibility.

5. **Preprocessing**  
   Toggle steps per model family: cleaning, encoding (OHE/CatBoost), feature filtering/selection, scaling/normalization, outlier clipping, class weights/resampling, time features. Every choice is captured in metadata.

6. **Training & Evaluation**  
   Fit baseline models and report **Accuracy, macro-F1, macro-Precision/Recall, ROC-AUC**, plus **training time, inference time, model size, feature count**. Visuals include confusion matrices, ROC/PR curves, and **Top-25 feature importance**.

7. **Compare (Raw vs. Preprocessed)**  
   Train with a **minimal** pipeline (only what’s needed to run) and with the **full** pipeline. See deltas for metrics and efficiency to quantify preprocessing benefits.

8. **Reporting & Export**  
   One click to **download**: cleaned datasets, splits, configs as JSON, trained models (PKL), and a **PDF report** assembling all figures, tables, and settings.

""")

