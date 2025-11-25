# functions
Reusable, unit-testable helper modules for A.R.C. (Automated Research for Cybersecurity).  
Keep Streamlit pages thin; put computation and orchestration here. Functions should return data/figures/objects for the UI to render.

## What belongs here
- Dataframe utilities (loading, coercions, column ops)
- Schema and label checks (duplicates, dtypes, class balance, drift)
- Splitting helpers (random/stratified/time-based; leakage guards)
- Preprocessing primitives (imputation, encoding, scaling, selection, clipping)
- Metrics and evaluation helpers (confusion matrix prep, ROC/PR)
- Plot builders that return figures (no Streamlit calls)
- Report assembly helpers for the PDF reporter

## Modules (overview)
- f00_VarInitialisation.py
  - Initialize st.session_state defaults and global flags used across pages
  - Provide sane defaults for dataset, label/timestamp columns, seeds
- f00_Sidebar.py
  - Build the sidebar UI
- f00_Logger.py
  - Lightweight run logger: show, save, clear
  - Consumed by the Reporter and Logger to render a log appendix
- f01_DataLoader.py
  - CSV/TXT loaders (single or multi-file), delimiter handling, header inference
  - Selective PCAP: flow/session helpers (if enabled)
  - Timestamp column detection
- f02_SchemaValidator.py
  - Schema validity checks (e.g. missing values)
  - Schema modification (e.g. renaming)
- f03_LabelValidator.py
  - Schema validity checks (e.g. class balance)
  - Label modification (renaming)
- f04_Splitter.py
  - Random, stratified, and time-based splits (train/val/test)
  - Quality checks (e.g. class distribution)
  - Split visualization helpers
- f05_Preprocessor.py
  - Toggleable preprocessing primitives (impute/encode/scale/select/clip)
- f06_Trainer_Evaluator.py
  - Baseline models orchestration (RF, GBM/CatBoost, SVM, MLP, LR)
  - Metrics (accuracy, macro-F1/precision/recall, ROC-AUC) and efficiency KPIs (train/infer time, model size, feature count)
  - Plots (confusion, ROC, PR) and minimal_raw_pipeline(...) vs full pipeline comparison
- f09_Reporter.py
  - ReportLab PDF assembly and section builders

## How pages should use these modules
- Collect inputs in the page (sidebar choices, paths, label/timestamp columns)
- Call functions from functions/ with plain Python/pandas arguments
- Render returned data/figures in the page
