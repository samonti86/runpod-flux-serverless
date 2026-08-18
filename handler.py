"""
Runpod serverless handler - FLUX.1-dev text-to-image.

Contract (Runpod serverless):
    A job arrives as {"input": {...}}. Whatever this function returns is JSON-
    serialised and sent back as the job output.

Two design decisions matter more than anything else in this file.

1. WHERE THE MODEL IS LOADED. It is loaded once at module scope, not inside
   handler(). A serverless worker is a long-lived process that serves many jobs:
   module scope runs once per worker boot (the "cold start"), handler() runs once
   per request. Loading a 34 GB model inside handler() would work, but every
   request would pay the load cost instead of only the first. This is the
   serverless equivalent of opening a database connection per query instead of
   using a pool.

2. WHERE THE HF CACHE LIVES. HF_HOME is pointed at a Runpod network volume when
   one is attached, so the download happens once for the whole endpoint rather
   than once per worker. This must be set BEFORE huggingface_hub is imported --
   it reads HF_HOME into module constants at import time, so setting it after
   the import silently does nothing.
"""

import os
import time

# -- Cache location (must precede the huggingface_hub import; see above) -------
_VOLUME = "/runpod-volume"          # Runpod mounts an attached network volume here
_DEFAULT_CACHE = os.path.join(_VOLUME, "huggingface") if os.path.isdir(_VOLUME) else "/models"
os.environ.setdefault("HF_HOME", _DEFAULT_CACHE)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import base64          # noqa: E402  (imports intentionally follow the env setup)
import io              # noqa: E402
import traceback       # noqa: E402

import runpod          # noqa: E402
import torch           # noqa: E402
from diffusers import FluxPipeline   # noqa: E402

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")

# Bounds. The endpoint is reachable once deployed, so input is treated as
# untrusted and every numeric field is clamped rather than passed through. An
# unclamped num_inference_steps is a simple way for a caller to hold a GPU open.
MAX_STEPS = 50
MAX_DIM = 1536
DIM_MULTIPLE = 16          # FLUX's VAE requires dimensions divisible by 16
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 3.5     # FLUX.1-dev is guidance-distilled; ~3.5 is its documented range
DEFAULT_DIM = 1024


# -- Cold start: runs once per worker, not once per request -------------------
if not (os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")):
    # Fail loudly and specifically. FLUX.1-dev is gated, so an anonymous fetch
    # returns a 401 that reads like a missing-repo error and wastes debugging time.
    raise RuntimeError(
        "HUGGING_FACE_HUB_TOKEN is not set. FLUX.1-dev is a gated model and cannot "
        "be downloaded anonymously. Set it on the endpoint under Environment "
        "variables as HUGGING_FACE_HUB_TOKEN referencing your Runpod secret."
    )

print("[boot] cache={} (network volume {})".format(
    os.environ["HF_HOME"],
    "attached" if os.path.isdir(_VOLUME) else "not attached"), flush=True)
print("[boot] loading {} - first boot downloads ~34 GB, later boots read from cache".format(
    MODEL_ID), flush=True)

_t0 = time.time()
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("[boot] ready in {:.1f}s".format(time.time() - _t0), flush=True)

if torch.cuda.is_available():
    print("[boot] gpu={} vram_allocated={:.1f}GB".format(
        torch.cuda.get_device_name(0),
        torch.cuda.memory_allocated() / 1e9), flush=True)


def _clamp(value, low, high, default):
    """Coerce a caller-supplied value into [low, high], falling back to default."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _round_to(value, multiple):
    """FLUX requires height/width divisible by 16; round rather than reject."""
    return max(multiple, (value // multiple) * multiple)


def handler(job):
    """Generate one image from a text prompt.

    Input:
        prompt              (str, required)
        negative_prompt     (str, optional)
        num_inference_steps (int, 1-50,   default 28)
        guidance_scale      (float,       default 3.5)
        width, height       (int, <=1536, default 1024, rounded to /16)
        seed                (int, optional - omit for random)

    Output:
        image_base64, seed, and the parameters actually used after clamping.
    """
    try:
        job_input = job.get("input") or {}

        prompt = job_input.get("prompt")
        if not prompt or not str(prompt).strip():
            # A structured error, not an exception: the caller gets an actionable
            # message and the worker stays warm for the next job.
            return {"error": "The 'prompt' field is required and cannot be empty."}
        prompt = str(prompt).strip()

        steps = _clamp(job_input.get("num_inference_steps"), 1, MAX_STEPS, DEFAULT_STEPS)
        width = _round_to(_clamp(job_input.get("width"), DIM_MULTIPLE, MAX_DIM, DEFAULT_DIM), DIM_MULTIPLE)
        height = _round_to(_clamp(job_input.get("height"), DIM_MULTIPLE, MAX_DIM, DEFAULT_DIM), DIM_MULTIPLE)

        try:
            guidance = float(job_input.get("guidance_scale", DEFAULT_GUIDANCE))
        except (TypeError, ValueError):
            guidance = DEFAULT_GUIDANCE
        guidance = max(0.0, min(20.0, guidance))

        # The seed is echoed back so any image can be reproduced exactly.
        seed = job_input.get("seed")
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        print("[job] steps={} {}x{} guidance={} seed={}".format(
            steps, width, height, guidance, seed), flush=True)
        t0 = time.time()

        result = pipe(
            prompt=prompt,
            negative_prompt=job_input.get("negative_prompt"),
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            max_sequence_length=512,
            generator=generator,
        )
        elapsed = time.time() - t0

        # Serverless responses are JSON, so the image goes back base64-encoded.
        # At higher volume the better pattern is writing to S3 and returning a
        # URL - base64 inflates the payload ~33% and every byte crosses the gateway.
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print("[job] done in {:.1f}s ({:.0f} KB base64)".format(
            elapsed, len(image_b64) / 1024), flush=True)

        return {
            "image_base64": image_b64,
            "seed": seed,
            "parameters": {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "width": width,
                "height": height,
                "model": MODEL_ID,
            },
            "generation_time_seconds": round(elapsed, 2),
        }

    except torch.cuda.OutOfMemoryError:
        # Caught separately because it is the most likely production failure and
        # the fix is caller-side (smaller dimensions) rather than a code bug.
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory. Try smaller width/height or fewer steps."}

    except Exception as exc:
        # Return the error rather than letting the worker die, so a failed job is
        # diagnosable from the Runpod dashboard.
        print(traceback.format_exc(), flush=True)
        return {"error": "{}: {}".format(type(exc).__name__, exc)}


runpod.serverless.start({"handler": handler})
