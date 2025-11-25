# ARC Code

This directory contains the complete source code required to run the A.R.C. pipeline, including the Streamlit application, backend functions, and Docker configuration.

---

## Structure

- ARC Code:
-- docker ignore
-- Dockerfile
-- requirements.txt
--ARC
--- .streamlit:
---- config.toml
--- functions
--- pages/
--- A_R_C.py

---

## Components

### .dockerignore
Specifies unneeded files excluded from the Docker build context.

### Dockerfile
Defines instructions to build a Docker image for the A.R.C. application.

### requirements.txt
Python package dependencies.

### ARC/
Contains the Streamlit application source code and structure:
- functions: backend logic
- pages: Streamlit UI modules
- .streamlit: configuration values (such as max upload size)
- A_R_C.py: the entry point of the application

---

## Running the App

pip install -r requirements.txt
streamlit run ARC/A_R_C.py

This starts the A.R.C. interface in the browser.