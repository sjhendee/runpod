# runpod/human/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import os, time

app = FastAPI(title="Mirism Human Pod")
START = time.time()
VERSION = os.getenv("APP_VERSION", "0.1.0")

# -------- models --------
class EchoIn(BaseModel):
    message: str

class ImageIn(BaseModel):
    imageUrl: str

# -------- core endpoints (under /api) --------
@app.get("/api/health")
def api_health():
    return {"ok": True, "service": "human", "version": VERSION, "uptime_s": round(time.time()-START, 3)}

@app.post("/api/echo")
def api_echo(body: EchoIn):
    return {"echo": body.message}

# simple stubs to unblock frontend quickly
@app.post("/api/pose")
def api_pose(body: ImageIn):
    # trivial fake keypoints; replace with real pose when ready
    return {
        "ok": True,
        "imageUrl": body.imageUrl,
        "keypoints": [{"x": 0.33, "y": 0.15}, {"x": 0.50, "y": 0.30}, {"x": 0.66, "y": 0.45}]
    }

@app.post("/api/segment")
def api_segment(body: ImageIn):
    # echo back a "mask" placeholder; replace with real segmentation
    return {"ok": True, "imageUrl": body.imageUrl, "maskUrl": body.imageUrl}

# -------- aliases (plain paths) to avoid client 404s --------
@app.get("/")
def root():
    return {"ok": True, "service": "human", "version": VERSION}

@app.get("/health")
def health():
    return api_health()

@app.post("/echo")
def echo(body: EchoIn):
    return api_echo(body)

@app.post("/pose")
def pose(body: ImageIn):
    return api_pose(body)

@app.post("/segment")
def segment(body: ImageIn):
    return api_segment(body)
