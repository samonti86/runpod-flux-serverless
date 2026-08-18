"""
Test client for the deployed Runpod serverless endpoint.

Usage:
    export RUNPOD_API_KEY=...
    export RUNPOD_ENDPOINT_ID=...
    python test_endpoint.py "a red fox in tall grass at golden hour"

Uses /runsync, which blocks until the job finishes and returns the result in one
response. The alternative, /run, returns a job id immediately and you poll
/status/<id> — better for long jobs or a UI, unnecessary here.
"""

import base64
import json
import os
import sys
import time
import urllib.request

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not API_KEY or not ENDPOINT_ID:
    sys.exit("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID first.")

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

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
)

print(f"POST {url}")
print(f"prompt: {prompt!r}")
print("waiting (a cold worker also loads ~34 GB of weights, so the first call is slower) ...")

t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as resp:
    body = json.load(resp)
wall = time.time() - t0

status = body.get("status")
print(f"\nstatus: {status}   round-trip: {wall:.1f}s")

if status != "COMPLETED":
    print(json.dumps(body, indent=2)[:2000])
    sys.exit(1)

out = body["output"]
if "error" in out:
    print("handler returned an error:", out["error"])
    sys.exit(1)

os.makedirs("output", exist_ok=True)
name = f"output/flux_{out['seed']}.png"
with open(name, "wb") as f:
    f.write(base64.b64decode(out["image_base64"]))

print(f"saved:   {name}")
print(f"seed:    {out['seed']}  (reuse it to reproduce this exact image)")
print(f"gpu time: {out.get('generation_time_seconds')}s of the {wall:.1f}s round-trip")
print(f"params:  {json.dumps(out['parameters'], indent=2)}")

# Delay metrics from Runpod itself — useful for separating queue wait and cold
# start from actual inference when explaining endpoint performance.
for k in ("delayTime", "executionTime"):
    if k in body:
        print(f"{k}: {body[k]} ms")
