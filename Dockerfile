# FLUX.1-dev serverless worker for Runpod.
#
# Built by Runpod from GitHub (Serverless -> New Endpoint -> import from GitHub),
# which builds on Runpod's infrastructure and stores the image in their private
# registry. The Hugging Face token is supplied as the build arg HF_TOKEN.
#
# Equivalent local build, which should use a secret mount because the image would
# be pushed to a public registry:
#
#   podman build --secret id=hf_token,src=$HOME/.hf_token #     -t docker.io/samontie86/flux-serverless:v1 .
#
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
# FLUX.1-dev is gated, so this download must authenticate or it 401s.
#
# The token is taken as a build ARG rather than a BuildKit secret mount, because
# a hosted builder may not support --mount=type=secret and an unparseable
# Dockerfile fails before it can report anything useful. The cost is that the
# value is recoverable from image history -- acceptable here because Runpod
# builds into its own private registry. For a build pushed to a PUBLIC registry,
# use the secret-mount form shown in the README instead.
ARG HF_TOKEN=""
ARG HUGGING_FACE_HUB_TOKEN=""
RUN HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}" python builder/download_model.py

# Handler last — cheap layer, fast to rebuild while iterating.
COPY handler.py test_input.json ./

CMD ["python", "-u", "handler.py"]
