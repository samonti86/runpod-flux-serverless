# FLUX.1-dev on Runpod Serverless

A serverless text-to-image endpoint: send a prompt, get a PNG back. Built for the
Runpod Serverless Endpoint case study.

**Model:** [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
· **Platform:** Runpod Serverless · **Base:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

---

## How it works

```
POST /run  ──►  Runpod queue  ──►  worker (scale-to-zero)
                                      │
                                      ├─ cold start: load FLUX.1-dev from local disk
                                      └─ per request: handler(job) ──► base64 PNG
```

The worker is a long-lived process. `handler()` runs once per request; everything
at module scope runs once per worker boot. **The model is loaded at module scope**
— see [`handler.py`](handler.py). Loading it inside `handler()` would make every
request pay the model-load cost instead of only the first.

## Repository layout

| File | Purpose |
|---|---|
| `handler.py` | The serverless handler: input validation, inference, base64 response |
| `Dockerfile` | Builds the worker image with the model baked in |
| `builder/download_model.py` | Downloads FLUX.1-dev at **build** time, not run time |
| `requirements.txt` | Pinned dependencies |
| `test_input.json` | Sample job payload (Runpod's local-test convention) |

## API

### Input

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | — | **Required** |
| `negative_prompt` | string | `null` | Optional |
| `num_inference_steps` | int | `28` | Clamped to 1–50 |
| `guidance_scale` | float | `3.5` | Clamped to 0–20 |
| `width` / `height` | int | `1024` | Clamped to ≤1536, rounded down to a multiple of 16 |
| `seed` | int | random | Echoed back, so any image is reproducible |

### Output

```json
{
  "image_base64": "iVBORw0KGgo...",
  "seed": 42,
  "parameters": { "prompt": "...", "num_inference_steps": 28, "...": "..." },
  "generation_time_seconds": 12.4
}
```

Errors return `{"error": "..."}` rather than raising, so a bad request fails the
job cleanly and leaves the worker warm for the next one.

## Build

FLUX.1-dev is a **gated** repo — the Hugging Face account behind the token must
have accepted the licence on the model page first.

```bash
export HF_TOKEN=hf_xxxxxxxx

docker buildx build --platform linux/amd64 \
  --secret id=hf_token,env=HF_TOKEN \
  -t <dockerhub-user>/flux-serverless:v1 --push .
```

> **On the token.** It is passed as a BuildKit **secret**, mounted only for the
> download step. It is never `COPY`d, never an `ENV`, and never becomes part of a
> layer — so it cannot be recovered from the published image with
> `docker history`. A build arg or an env var would leave it in the image
> permanently, which matters here because this image is public.

The image is roughly **41 GB** (≈34 GB of weights + CUDA/PyTorch base). That is the
cost of the case study's requirement to include the model in the image, and it is
the right trade: a scale-to-zero endpoint that downloaded 34 GB on every cold
start would not be usable.

## Deploy

1. Runpod console → **Serverless** → **New Endpoint**
2. Source: the Docker image above
3. GPU: **48 GB** class (L40S / A6000). FLUX.1-dev in bf16 is ~24 GB of weights
   plus activations; 24 GB cards are not enough headroom at 1024×1024.
4. Container disk large enough for the image
5. Workers: min 0 (scale to zero), max 1 for testing

## Test

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json | python -c "import sys,json,base64; \
      d=json.load(sys.stdin); \
      open('output.png','wb').write(base64.b64decode(d['output']['image_base64'])); \
      print('wrote output.png, seed=', d['output']['seed'])"
```

## Notes and trade-offs

- **Base64 vs object storage.** Returning the image inline keeps the endpoint
  self-contained and easy to test. At higher volume or larger resolutions the
  better pattern is to write to S3 and return a URL — base64 inflates the payload
  by ~33% and every byte crosses the API gateway.
- **Cold starts.** With weights on local disk a cold worker is ready in roughly
  20–40s. Runpod's *active workers* setting removes this at the cost of paying for
  idle GPU; for bursty traffic, scale-to-zero plus a warm-up request is usually the
  better trade.
- **Input clamping.** The endpoint is public once deployed. `num_inference_steps`
  is clamped because an unbounded value is an easy way for a caller to occupy a
  worker — and a GPU bill — indefinitely.
- **bf16, not fp16.** FLUX is released in bfloat16; fp16 overflows on this
  architecture and produces black images.

## Authorship

Written by me with AI coding assistance (Claude), which is how I build tooling —
I specify what needs to exist, drive the assistant, then read, test and debug the
result. Every design decision documented here is one I can defend and did decide:
where the model loads, why the token is a build secret, why inputs are clamped,
and why the weights are baked into the image rather than fetched at runtime.
