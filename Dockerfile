# CPU-only image. Model and dataset are fetched at build time so the container runs offline.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /app

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch the pinned BERT revision and the pinned StereoSet revision.
COPY config.yaml .
RUN python - <<'PY'
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
cfg = yaml.safe_load(open("config.yaml"))
snapshot_download(cfg["model_name"], revision=cfg["model_revision"],
                  allow_patterns=["*.json", "*.txt", "*.safetensors"])
load_dataset(cfg["dataset_name"], cfg["dataset_config"],
             split=cfg["dataset_split"], revision=cfg["dataset_revision"])
PY

COPY src/ src/

# Run as an unprivileged user; uid 1000 so bind-mounted data/ and results/ stay owned by the host user.
RUN useradd --uid 1000 --create-home app && mkdir -p data results && chown -R app /app /opt/hf
USER app

ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["sh", "-c", "python src/data.py && python src/embed.py && python src/train.py && python src/baselines.py"]
