FROM python:3.11-slim
WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
ENV MPLBACKEND=Agg PYTHONHASHSEED=0
