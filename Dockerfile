FROM python:3.11-slim
WORKDIR /work
COPY requirements.txt /tmp/severity-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/severity-requirements.txt
