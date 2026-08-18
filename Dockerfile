# syntax=docker/dockerfile:1.7
#
# FLUX.1-dev serverless worker for Runpod.
#
# Build (note --secret: the HF token is mounted for one RUN step and is never
# written to a layer, so it cannot be recovered with `docker history`):
#
#   export HF_TOKEN=hf_xxx
#   docker buildx build --platform linux/amd64 \
#     --secret id=hf_token,env=HF_TOKEN \
#     -t <dockerhub-user>/flux-serverless:v1 --push .

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# Dependencies first, so editing handler.py later does not invalidate the
# (very expensive) model layer below.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── The expensive layer: ~34 GB of weights baked into the image ──────────────
COPY builder/download_model.py builder/download_model.py
RUN --mount=type=secret,id=hf_token,required=true \
    HUGGING_FACE_HUB_TOKEN="$(cat /run/secrets/hf_token)" \
    python builder/download_model.py

# Handler last — cheap layer, fast to rebuild while iterating.
COPY handler.py test_input.json ./

CMD ["python", "-u", "handler.py"]
