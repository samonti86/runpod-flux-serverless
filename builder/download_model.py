"""
Bake FLUX.1-dev into the image at build time.

The case study asks for "a Docker image that includes your serverless handler
AND the model", so the weights are downloaded here rather than fetched on first
request. The trade-off is deliberate:

    baked in   -> ~41 GB image, slow to push once, but cold start is just
                  loading from local disk (~20-40s)
    downloaded -> small image, but every cold worker pulls ~34 GB from
                  Hugging Face before it can serve anything

For a scale-to-zero endpoint the second option makes cold starts unusable, which
is why the instruction to bake it in is the right one.

FLUX.1-dev ships the same weights twice: once as single-file checkpoints
(flux1-dev.safetensors, ae.safetensors) for ComfyUI-style loaders, and once in
the sharded diffusers layout that FluxPipeline.from_pretrained expects. Pulling
both would add ~24 GB of files this image never opens, so the single-file
variants are excluded.
"""

import os
import sys

from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")

# FLUX.1-dev is a gated repo: the token must belong to an account that has
# accepted the licence on the model page, or this 401s.
token = (os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
if not token:
    sys.exit(
        "No Hugging Face token found. FLUX.1-dev is a gated repo, so the build "
        "cannot download it anonymously. Provide the token as a BuildKit secret "
        "(id=hf_token), a build arg (HF_TOKEN), or a build environment variable."
    )
print(f"[build] token present (…{token[-4:]}), authenticating", flush=True)

print(f"[build] downloading {MODEL_ID} (diffusers layout only) ...", flush=True)

path = snapshot_download(
    repo_id=MODEL_ID,
    token=token,
    ignore_patterns=[
        "flux1-dev.safetensors",   # single-file duplicate of transformer/ (~23.8 GB)
        "ae.safetensors",          # single-file duplicate of vae/
        "*.gguf",                  # quantised community variants
        "*.jpg", "*.png",          # sample images from the model card
    ],
)

print(f"[build] model cached at {path}", flush=True)

total = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(path)
    for f in files
    if not os.path.islink(os.path.join(root, f))
)
print(f"[build] on-disk size: {total / 1e9:.1f} GB", flush=True)
