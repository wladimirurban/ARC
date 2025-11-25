# ARC — Application Source Code

This directory contains the core implementation of the A.R.C. Streamlit application, including the backend logic, UI pages, and configuration.

---

## Structure
--ARC
--- .streamlit/
--- functions/
--- pages/
--- A_R_C.py

---

## Overview of Subdirectories

### .streamlit
Holds Streamlit configuration files such as maximum upload size and UI preferences.

### functions
Contains all backend logic:
- Data loading and parsing
- Schema and label validation
- Splitting and preprocessing
- Model training and evaluation
- PDF report generation

### pages
Contains Streamlit user interface pages, each representing a module in the pipeline.

### A_R_C.py
The main entry point of the application.  
Initializes session variables, sets up navigation, and displays the introductory overview.

---

## Design Notes

- Complete separation of UI (pages/) and logic (functions/)
- All module states stored in `streamlit.session_state`
- Consistent naming conventions and modular extension possibilities  
- Designed for reproducible research and dataset preparation workflows