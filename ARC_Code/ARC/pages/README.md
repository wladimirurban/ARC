# pages

Streamlit UI pages for the A.R.C. (Automated Research for Cybersecurity).  
Each file in this folder becomes a page in the app’s sidebar. Pages are thin orchestration layers: they collect inputs, call logic from functions/, and render outputs.

## What belongs here
- Page entrypoints such as 01_Data_Loader.py, 02_Schema_Validator.py, …
- Minimal UI code: sidebar widgets, layout, status messages, charts/tables
- Calls into functions/ for all non-UI work (loading, checks, splits, training, report export)

## Recommended pages (typical flow)
- 01_Data_Loader.py
  Upload data (CSV/TXT/JSON, optional PCAP→flows), set label/timestamp columns, quick preview.
- 02_Schema_Validator.py
  Dtype coercions, duplicate detection, temporal integrity checks, sparsity/features overview.
- 03_Label_Validator.py
  Class balance, entropy, rare/dominant classes, temporal drift and label normalization options.
- 04_Splitter.py
  Random/stratified/time-based splits and class distribution per split, unseen-label checks.
- 05_Preprocess.py
  Toggleable preprocessing steps (impute/encode/scale/select/clip/resample) with presets.
- 06_Train_Eval.py
  Baseline models; metrics (accuracy, macro-F1/precision/recall, ROC-AUC); efficiency KPIs; plots.
- 07_Compare.py
  Minimal raw vs. full preprocessing comparison; deltas for metrics and efficiency.
- 08_Report.py
  Assemble PDF; export artifacts (CSV/JSON/PKL/figures/tables); include run log.

> Use numeric prefixes (01_, 02_, …) to keep navigation order stable.

## Page design guidelines
- Keep logic in functions/. Pages should prepare inputs, call helpers, and render results.
- Use st.session_state for cross-page state. Prefer stable keys with a common prefix.
- Be deterministic: pass seeds and cutoffs explicitly to functions/.
- Make errors visible and actionable with st.error, not hidden exceptions.