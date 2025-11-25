# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-tk \
        libgomp1 \
        ca-certificates \
        build-essential \
        fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# make sure the file is really named requirements.txt in ARC/
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# IMPORTANT: match the exact case: StreamLit
COPY ARC/ /app/ARC/

EXPOSE 8501

# absolute path so WORKDIR doesn't matter
CMD ["streamlit","run","/app/ARC/A_R_C.py","--server.address=0.0.0.0","--server.port=8501","--browser.gatherUsageStats=false"]