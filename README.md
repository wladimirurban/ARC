# A.R.C. — Automated Research for Cybersecurity

A.R.C. (Automated Research for Cybersecurity) is a modular end-to-end pipeline for preparing, validating, preprocessing, splitting, training, evaluating, and comparing machine-learning workflows for cybersecurity datasets. It is designed to deliver reproducible, auditable, and standardized data preparation and evaluation for research.

The system provides:

- A Streamlit application for interactive dataset analysis
- A reproducible Docker environment
- Modular Python backend functions
- Automated PDF reporting and export tools
- Full dataset preparation workflow: load → validate → split → preprocess → train → compare → export

---

## Repository Structure
- Dokuments:
-- Thesis
-- Dokumentation
-- Powerpoint presenations

- Data:
-- DNP3 dataset
-- CIC-IIoT dataset

- Docker Image
-- Docker Image of the project (arc)

- ARC Code:
-- docker ignore
-- Dockerfile
-- requirements.txt
-- ARC
--- .streamlit:
---- config.toml (sets the dataset file size)
--- functions (holds all function files)
--- pages (holds all pages / UI files)
--- A_R_C.py (Start point)

---

## Running the Application

### Using Docker
docker load -i “Docker Image/arc.tar”
docker run -p 8501:8501 arc
Open http://localhost:8501

### Running Locally
cd “ARC Code”
pip install -r requirements.txt
streamlit run ARC/A_R_C.py

---

## Functional Overview

- Load CSV/TXT/JSON or parse PCAP/PCAPNG files
- Validate schema: datatypes, timestamps, duplicates, sparsity, granularity
- Validate labels: entropy, rare/dominant classes, inconsistencies, drift
- Random, Stratified, or Time-based splitting
- Preprocessing: cleaning, encoding, selection, scaling, PCA, imbalance handling, time features
- Model training: Random Forest, Gradient Boosting, SVM, MLP, Logistic Regression
- Raw vs. Preprocessed comparison
- Export cleaned datasets, metadata, trained models, metrics, and PDF reports

---

## Contributing

Contributions are welcome. The project is modular, and new functions or UI modules can be added easily.
