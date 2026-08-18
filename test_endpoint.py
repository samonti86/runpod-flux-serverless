"""
Test client for the deployed Runpod serverless endpoint.

Usage:
    export RUNPOD_API_KEY=...
    export RUNPOD_ENDPOINT_ID=...
    python test_endpoint.py "a red fox in tall grass at golden hour"

On /runsync vs /run:

    /runsync is only *mostly* synchronous. It blocks for up to about 90 seconds
    and, if the job has not finished by then, returns the job handle with
    status IN_QUEUE instead of the result. That is easy to mistake for a
    failure -- the job is still running perfectly well.

    So this client submits to /runsync for the fast path (a warm worker returns
    in seconds) and falls back to polling /status/<id> when the job outlives the
    synchronous window, which is what happens on a cold start.
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not API_KEY or not ENDPOINT_ID:
    sys.exit("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID first.")

BASE = "https://api.runpod.ai/v2/{}".format(ENDPOINT_ID)
HEADERS = {"Authorization": "Bearer {}".format(API_KEY), "Content-Type": "application/json"}
POLL_TIMEOUT = 900          # seconds to keep polling before giving up
POLL_INTERVAL = 5

prompt = " ".join(sys.argv[1:]) or \
    "a red fox sitting in tall grass at golden hour, shallow depth of field, photorealistic"

payload = {
    "input": {
        "prompt": prompt,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024,
        "seed": 42,
    }
}


def call(url, data=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers=HEADERS,
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


print("POST {}/runsync".format(BASE))
print("prompt: {!r}".format(prompt))
print("submitting (a cold worker loads ~34 GB of weights first) ...")

t0 = time.time()
body = call(BASE + "/runsync", payload)
job_id = body.get("id")
status = body.get("status")

# Cold start: the job outlived the synchronous window, so poll for it.
while status in ("IN_QUEUE", "IN_PROGRESS"):
    if time.time() - t0 > POLL_TIMEOUT:
        sys.exit("Gave up after {}s with status {}".format(POLL_TIMEOUT, status))
    print("  {:>5.0f}s  {}".format(time.time() - t0, status), flush=True)
    time.sleep(POLL_INTERVAL)
    body = call("{}/status/{}".format(BASE, job_id))
    status = body.get("status")

wall = time.time() - t0
print("\nstatus: {}   round-trip: {:.1f}s".format(status, wall))

if status != "COMPLETED":
    print(json.dumps(body, indent=2)[:2000])
    sys.exit(1)

out = body["output"]
if "error" in out:
    print("handler returned an error:", out["error"])
    sys.exit(1)

os.makedirs("output", exist_ok=True)
name = "output/flux_{}.png".format(out["seed"])
with open(name, "wb") as f:
    f.write(base64.b64decode(out["image_base64"]))

print("saved:    {}".format(name))
print("seed:     {}  (reuse it to reproduce this exact image)".format(out["seed"]))
print("gpu time: {}s of the {:.1f}s round-trip".format(
    out.get("generation_time_seconds"), wall))
print("params:   {}".format(json.dumps(out["parameters"], indent=2)))

# Runpod's own timings. delayTime is queue wait plus cold start; executionTime is
# the handler itself. Keeping them separate is what lets you say whether an
# endpoint is slow because inference is slow or because it was scaled to zero.
for k in ("delayTime", "executionTime"):
    if k in body:
        print("{}: {} ms".format(k, body[k]))
