"""
Runpod serverless handler — FLUX.1-dev text-to-image.

Contract (Runpod serverless):
    A job arrives as {"input": {...}}. Whatever this function returns is JSON-
    serialised and sent back as the job output. Anything raised is reported as
    a failed job.

The single most important design decision in this file is WHERE the model is
loaded. It is loaded once at module scope (below), not inside handler().

Why that matters: a Runpod serverless worker is a long-lived process that
handles many jobs. Module scope runs once when the worker boots — the "cold
start". handler() runs once per request. Loading a 24 GB model inside handler()
would work, but every single request would pay the ~20-40s load cost instead of
just the first one. This is the serverless equivalent of opening a database
connection per query instead of using a pool.
"""

import base64
import io
import os
import time
import traceback

import runpod
import torch
from diffusers import FluxPipeline

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")

# Bounds. The endpoint is public once deployed, so inputs are treated as
# untrusted: every numeric field is clamped rather than passed through. An
# unclamped num_inference_steps is a straightforward way for a caller to run up
# a GPU bill or hold a worker open indefinitely.
MAX_STEPS = 50
MAX_DIM = 1536
DIM_MULTIPLE = 16          # FLUX's VAE requires dimensions divisible by 16
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 3.5     # FLUX.1-dev is guidance-distilled; ~3.5 is the documented sweet spot
DEFAULT_DIM = 1024


# ── Cold start ────────────────────────────────────────────────────────────────
# Runs once per worker, not once per request.
print(f"[boot] loading {MODEL_ID} ...", flush=True)
_t0 = time.time()

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # FLUX is released in bf16; fp16 overflows on this architecture
)
pipe.to("cuda")

print(f"[boot] model ready in {time.time() - _t0:.1f}s", flush=True)
if torch.cuda.is_available():
    print(f"[boot] gpu={torch.cuda.get_device_name(0)} "
          f"vram_allocated={torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)


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
        num_inference_steps (int, 1-50,    default 28)
        guidance_scale      (float,        default 3.5)
        width, height       (int, <=1536,  default 1024, rounded to /16)
        seed                (int, optional — omit for random)

    Output:
        image_base64, seed, and the parameters actually used after clamping.
    """
    try:
        job_input = job.get("input") or {}

        prompt = job_input.get("prompt")
        if not prompt or not str(prompt).strip():
            # A structured error, not an exception. The caller gets an
            # actionable message and the worker stays warm for the next job.
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

        # Seed is echoed back so a caller can reproduce any image exactly.
        seed = job_input.get("seed")
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"[job] steps={steps} {width}x{height} guidance={guidance} seed={seed}", flush=True)
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
        # For large or high-volume payloads the better pattern is to write to S3
        # and return a URL — noted in the README as the production alternative.
        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print(f"[job] done in {elapsed:.1f}s ({len(image_b64) / 1024:.0f} KB base64)", flush=True)

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
        # Worth catching separately: it is the most likely production failure,
        # and the fix is a caller-side one (smaller dimensions) rather than a bug.
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory. Try smaller width/height or fewer steps."}

    except Exception as exc:
        # Return the traceback rather than letting the worker die, so a failed
        # job is diagnosable from the Runpod dashboard.
        print(traceback.format_exc(), flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})
