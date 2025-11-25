# Docker Image

This directory contains the prebuilt Docker image archive of the full A.R.C. application.

---

## Contents

- DockerImage
-- arc.tar

---

## Running the Docker Image

### 1. Load the image

docker load -i arc.tar

### 2. Start the container

docker run -p 8501:8501 arc

### 3. Access the application

Open http://localhost:8501

---

## Why Use Docker

- Ensures a reproducible execution environment
- No need to install Python or dependencies locally
- Easy to deploy to servers or cloud platforms
- Guaranteed compatibility with the Streamlit UI and backend functions