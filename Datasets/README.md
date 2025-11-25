# Data

This folder contains the cybersecurity datasets used for evaluation and demonstration within the A.R.C. pipeline.

These datasets are used as input for the Data Loader module. A.R.C. will process and export validated and preprocessed variants separately.

---

## Structure

- Data:
-- DNP3 dataset
-- CIC-IIoT dataset

---

## DNP3 Dataset

Industrial-control (ICS/SCADA) protocol dataset featuring labeled traffic with temporal information and several attack types.

Contains:

- Normal and malicious traffic
- Well-defined timestamp fields
- Useful for evaluating time-based validation and splitting

---

## CIC-IIoT Dataset

A comprehensive multi-attack IoT/Industrial dataset built by CIC.

Contains:

- Many categorical and numerical features
- High imbalance between classes
- Several challenging label distributions

---

## Usage Notes

These datasets should be loaded through the A.R.C. Data Loader.  
A.R.C. supports automatic timestamp detection, label selection, schema and label validation, splitting, and processing.
