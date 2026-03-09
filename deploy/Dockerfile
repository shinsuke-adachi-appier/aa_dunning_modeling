# Build from the deploy folder only (this directory). Upload this folder to Cloud Build / Cloud Run.
# No parent repo: all code is under deploy/ (lib/, inference_job/, trigger_job/).

FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements-deploy.txt

# Cloud Run Job: override command to either:
#   python inference_job/main.py
#   python trigger_job/main.py
