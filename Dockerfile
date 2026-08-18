# FLUX.1-dev serverless worker for Runpod.
#
# The weights are NOT baked into this image, and that is a deliberate decision
# forced by the platform. See README "Why the model is not in the image" -- the
# short version is that Runpod's GitHub builder invokes `docker buildx build`
# with neither --build-arg nor --secret, so a GATED model such as FLUX.1-dev
# cannot be authenticated for during a Runpod-side build. Proven from build logs,
# not assumed.
#
# The model is fetched once at worker cold start using the HUGGING_FACE_HUB_TOKEN
# secret configured on the endpoint, and cached on a Runpod network volume when
# one is attached so that subsequent workers start warm.
#
# Build and push:
#   podman build -t docker.io/samontie86/flux-serverless:v1 .
#   podman push docker.io/samontie86/flux-serverless:v1

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Dependencies are the only heavy layer now, so the image is ~4 GB to push
# rather than ~41 GB.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py test_input.json ./

CMD ["python", "-u", "handler.py"]
