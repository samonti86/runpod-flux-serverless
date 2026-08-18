# FLUX.1-dev on Runpod Serverless

A serverless text-to-image endpoint: send a prompt, get a PNG back.

**Model:** [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
· **Platform:** Runpod Serverless (Queue) · **Base:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

---

## How it works

```
POST /run (async) ──►  Runpod queue  ──►  worker (scale-to-zero)
                                          │
                                          ├─ cold start: load FLUX.1-dev
                                          └─ per request: handler(job) ──► base64 PNG
```

The worker is a long-lived process. `handler()` runs once per request; everything
at module scope runs once per worker boot. **The model is loaded at module scope**
— see [`handler.py`](handler.py). Loading it inside `handler()` would make every
request pay the model-load cost instead of only the first.

## Repository layout

| File | Purpose |
|---|---|
| `handler.py` | The serverless handler: cache setup, input validation, inference, base64 response |
| `Dockerfile` | Builds the worker image |
| `requirements.txt` | Pinned dependencies |
| `test_input.json` | Sample job payload |
| `test_endpoint.py` | Test client — calls `/runsync` and writes the PNG to disk |

---

## Why the model is not baked into the image

The case study asks for *"a Docker image that includes your serverless handler and
the model."* I built it that way first. **It cannot be built on Runpod, and I have
the build logs to prove why rather than assuming it.**

FLUX.1-dev is a **gated** model — downloading it requires an authenticated
Hugging Face token. So the build needs a secret. Two attempts, two failures:

**Attempt 1 — BuildKit secret mount** (`RUN --mount=type=secret,id=hf_token`):

```
#15 ERROR: secret hf_token: not found
ERROR: failed to build: failed to solve: secret hf_token: not found
```

**Attempt 2 — build argument** (`ARG HF_TOKEN`), with `HUGGING_FACE_HUB_TOKEN`
set as an endpoint environment variable backed by a Runpod secret:

```
#12 [6/7] RUN HUGGING_FACE_HUB_TOKEN="" python builder/download_model.py
#12 0.208 No Hugging Face token found. FLUX.1-dev is a gated repo...
```

Note the expansion: `HUGGING_FACE_HUB_TOKEN=""`. The build arg was empty, meaning
the endpoint's environment variables are **not** injected into the build.

The decisive evidence is the build command Runpod runs, which its own logs print:

```
/usr/bin/docker buildx build -t registry.runpod.net/samonti86-runpod-flux-serverless-...
  --file .../Dockerfile --network=default --progress=plain
  --output type=oci,dest=... --ulimit cpu=1800 --ulimit nofile=1024:1024
  --cache-to type=local,dest=...
```

**No `--build-arg`. No `--secret`.** Neither mechanism is available, so a gated
model cannot be authenticated for during a Runpod-side build. This is a platform
constraint, not a configuration mistake.

The alternative is building locally and pushing. I started that — the model downloaded
fine — but abandoned it partway through committing the layer: a ~41 GB image plus the
scratch space to compress it for upload exceeded the free disk on this machine, and at a
measured 66 Mbps upstream the push alone is ~80 minutes per iteration.

**What this repository does instead:** the image carries the handler and its
dependencies (6.5 GB uncompressed, 3.44 GB pushed), and the model is fetched once at
worker cold start using
the `HUGGING_FACE_HUB_TOKEN` secret, cached on a Runpod **network volume** so
subsequent workers start from cache rather than re-downloading. That is Runpod's
intended mechanism for large and gated models — the console exposes it directly
as the **Cached model** field.

The trade-off, stated plainly: **the first cold start against an empty volume is
slow** (~34 GB download). Every one after that reads from the volume, including on
later deployments. Baking the weights in would make that first start as fast as the
rest, at the cost of a ~41 GB image that Runpod's own builder cannot produce for a
gated model.

---

## API

### Input

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | **Required** |
| `num_inference_steps` | int | `28` | Clamped to 1–50 |
| `guidance_scale` | float | `3.5` | Clamped to 0–20 |
| `width` / `height` | int | `1024` | Clamped to ≤1536, rounded down to a multiple of 16 |
| `seed` | int | random | Echoed back, so any image is reproducible |

There is deliberately no `negative_prompt`. FLUX.1-dev is guidance-distilled, so
the classifier-free guidance pass a negative prompt would steer was trained out;
`FluxPipeline` raises `TypeError` if the argument is passed at all. Steer away
from unwanted content inside the prompt instead.

Inputs are clamped rather than trusted — an unbounded `num_inference_steps` is an
easy way for a caller to hold a GPU open indefinitely.

### Output

```json
{
  "image_base64": "iVBORw0KGgo...",
  "seed": 42,
  "parameters": { "prompt": "...", "num_inference_steps": 28, "...": "..." },
  "generation_time_seconds": 12.4
}
```

Errors return `{"error": "..."}` rather than raising, so a bad request fails
cleanly and leaves the worker warm for the next job.

---

## Build and push

```bash
podman build -t docker.io/samontie86/flux-serverless:v1 .
podman push docker.io/samontie86/flux-serverless:v1
```

No secret is needed at build time — that is the point of the design above.

## Deploy

Runpod console → **Serverless → New Endpoint → import from GitHub**, selecting this
repository, branch `master`, Dockerfile path `/Dockerfile`. Runpod builds the image
on its own infrastructure and stores it in its private registry — which works here
precisely because the build needs no secret.

| Setting | Value | Why |
|---|---|---|
| Endpoint type | **Queue** | Matches the `handler()` contract; Load balancer is for workers running their own HTTP server |
| GPU | **48 GB minimum** (A40 / A6000 / L40S); an **H100 80 GB** was what supply offered | Measured at run time: `vram_allocated=33.8GB`. 16 GB and 24 GB cards OOM during load |
| Active workers | 0 | Scale to zero — pay only while generating |
| Idle timeout | 120 s | Long enough that consecutive requests reuse a warm worker |
| Execution timeout | 600 s | Safety net; a generation is 10–20 s |
| FlashBoot | enabled | Free, and cuts cold starts |
| Env var | `HUGGING_FACE_HUB_TOKEN` = `{{ RUNPOD_SECRET_hf_token }}` | Gated model; referencing a secret keeps the token out of plain config |
| Network volume | attached | Caches the weights across workers |

> The Runpod quickstart suggests a 16 GB GPU. That is correct for its example
> handler, which loads no model. Sizing here follows the model, not the tutorial.

## Test

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
python test_endpoint.py "a red fox in tall grass at golden hour"
```

Writes the PNG to `output/`, prints the seed, and reports Runpod's own
`delayTime` and `executionTime` so queue wait and cold start can be separated
from actual inference.

The endpoint can also be exercised from the console's **Requests** tab or from
Postman by POSTing `test_input.json` to
`https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`.

---

## Measured results

Deployed on an **H100 80 GB SXM** in `US-CA-2`, with a 50 GB network volume
attached. All figures are from the endpoint's own logs, not estimates.

### Generation

![FLUX.1-dev output: a red fox in tall grass at golden hour](docs/example-output.png)

*Prompt: "a red fox sitting in tall grass at golden hour, shallow depth of field,
photorealistic" · 28 steps · guidance 3.5 · seed 42 · 1024x1024*

```
[job] steps=28 1024x1024 guidance=3.5 seed=42
28/28 denoising steps @ 4.09 it/s
[job] done in 7.8s (1488 KB base64)
```

**7.8 seconds** for a 1024x1024 image at 28 steps. VRAM in use: **33.8 GB** —
which is why a 16 GB or 24 GB GPU is not an option for this model in bf16.

### Cold start, and why there are three numbers

| Boot | Worker | Time | What it measures |
|---|---|---|---|
| 1st | `t5c3xw34j0crqm` | **177.3s** | Cold. Downloading 34 GB (23 files in 2m06s) |
| 2nd | `t5c3xw34j0crqm` | **8.9s** | Same worker restarting, OS page cache still warm |
| 3rd | `f2jgqs4h4xsqvp` | **41.1s** | **A different worker**, reading from the network volume |

**41.1s is the honest steady-state figure.** The 8.9s is flattering but not
representative — it is the same machine restarting with the weights still in RAM.

The third row is the one that validates the design: a worker that never
downloaded the model booted in 41s because the volume already held it. Without
the volume, every cold worker would repeat the 177s download.

### End-to-end, measured from the client

```
status: COMPLETED      round-trip: 29.5s
delayTime:      20,728 ms   <- queue wait + worker spin-up
executionTime:   8,198 ms   <- the handler
generation:       7.44 s    <- denoising alone
```

Seventy percent of that round trip was acquiring a worker, not generating. Reporting
"30 seconds per image" would be wrong and would point optimisation at the wrong
layer -- which is why `test_endpoint.py` prints Runpod's `delayTime` and
`executionTime` separately rather than just the wall clock.

### Cost

Billed per millisecond at $0.00133/s ($4.79/hr) with scale-to-zero, so the
endpoint costs nothing between requests. A cold start plus one generation is
roughly $0.07.

---

## Notes and trade-offs

- **bf16, not fp16.** FLUX is released in bfloat16; fp16 overflows on this
  architecture and yields black images.
- **Base64 vs object storage.** Returning the image inline keeps the endpoint
  self-contained and easy to test. At higher volume, writing to S3 and returning a
  URL is better — base64 inflates the payload ~33%.
- **`HF_HOME` is set before `huggingface_hub` is imported.** It is read into module
  constants at import time, so setting it afterwards silently does nothing. This is
  the kind of ordering bug that produces a correct-looking config and a full
  re-download every boot.
- **Cold start vs idle cost.** Runpod's *active workers* setting removes cold starts
  entirely at the cost of a permanently billed GPU. For bursty traffic, scale-to-zero
  plus FlashBoot and a warm-up request is usually the better trade.

## Authorship

Written by me with AI coding assistance, which is how I build tooling — I specify
what needs to exist, drive the assistant, then read, test and debug the result.
The design decisions documented here are ones I made and can defend: where the
model loads, why `HF_HOME` precedes the import, why inputs are clamped, why the
GPU is sized to the model rather than the quickstart, and why the weights are
fetched at cold start rather than baked in.
