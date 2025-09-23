# runpod/human/app.py-092325
from fastapi import FastAPI
from pydantic import BaseModel
import os, time
from typing import Optional
from io import BytesIO
import base64
import requests
from PIL import Image, ImageOps
try:
    from .detector import detect_items, extract_best_garment
except Exception:
    detect_items = None  # type: ignore
    extract_best_garment = None  # type: ignore

app = FastAPI(title="Mirism Human Pod")
START = time.time()
VERSION = os.getenv("APP_VERSION", "0.1.0")

# -------- models --------
class EchoIn(BaseModel):
    message: str

class ImageIn(BaseModel):
    imageUrl: str

class DetectIn(BaseModel):
    imageUrl: str
    maxItems: int | None = 8

class GarmentExtractIn(BaseModel):
    imageUrl: str
    category_hint: Optional[str] = None

# -------- core endpoints (under /api) --------
@app.get("/api/health")
def api_health():
    return {"ok": True, "service": "human", "version": VERSION, "uptime_s": round(time.time()-START, 3)}


if os.getenv("RUNPOD_ENABLE_PING", "1").lower() not in ("0", "false", "no", "off"):
    @app.get("/ping")
    def ping():
        """RunPod's load balancer probes /ping; respond with 200 so workers stay healthy."""
        return {"ok": True, "service": "human", "version": VERSION}

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

@app.post("/api/detect")
def api_detect(body: DetectIn):
    if detect_items is None:
        labels = ["Top","Sweater","Pants","Skirt/Dress","Shoes","Bag","Accessory"][: max(1, min(body.maxItems or 8, 8))]
        return {"ok": True, "items": [{"label": l} for l in labels]}
    try:
        items = detect_items(body.imageUrl, max_items=body.maxItems or 8)
        return {"ok": True, "items": items}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _data_url(pil: Image.Image) -> str:
    buf = BytesIO()
    pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"

@app.post("/api/garment/extract")
def api_garment_extract(body: GarmentExtractIn):
    if extract_best_garment is None:
        # Fallback simple full mask
        try:
            r = requests.get(body.imageUrl, timeout=10)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert('RGB')
            mask = Image.new('L', img.size, color=255)
            mask_url = _data_url(ImageOps.autocontrast(mask))
            return {"ok": True, "maskUrl": mask_url, "category": body.category_hint or "upper"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    try:
        res = extract_best_garment(body.imageUrl, category_hint=body.category_hint)
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}

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

@app.post("/detect")
def detect(body: DetectIn):
    return api_detect(body)

@app.post("/garment/extract")
def garment_extract(body: GarmentExtractIn):
    return api_garment_extract(body) 
