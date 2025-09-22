import os
from typing import List, Dict, Optional, Tuple
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
import requests

# Optional heavy deps; import lazily if present
try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection  # type: ignore
except Exception:  # pragma: no cover
    OwlViTProcessor = None  # type: ignore
    OwlViTForObjectDetection = None  # type: no cover

try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
except Exception:
    sam_model_registry = None  # type: ignore
    SamPredictor = None  # type: ignore

CLOTHING_PROMPTS = os.getenv("DETECT_PROMPTS", "shirt,sweater,blouse,top,jeans,pants,skirt,dress,coat,jacket,blazer,shoes,sneakers,heels,bag,handbag,scarf,hat").split(",")

_owl = None
_sam = None


def _load_owl() -> Optional[Tuple[OwlViTProcessor, OwlViTForObjectDetection]]:
    global _owl
    if _owl is not None:
        return _owl
    if OwlViTProcessor is None:
        return None
    try:
        proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        _owl = (proc, model)
        return _owl
    except Exception:
        return None


def _load_sam() -> Optional[SamPredictor]:
    global _sam
    if _sam is not None:
        return _sam
    if sam_model_registry is None:
        return None
    ckpt = os.getenv("SAM_CHECKPOINT_PATH")
    if not ckpt or not os.path.exists(ckpt):
        return None
    try:
        sam = sam_model_registry.get("vit_h")(checkpoint=ckpt)
        _sam = SamPredictor(sam)
        return _sam
    except Exception:
        return None


def _to_data_url(img: Image.Image) -> str:
    bio = BytesIO()
    img.save(bio, format="PNG")
    import base64
    return f"data:image/png;base64,{base64.b64encode(bio.getvalue()).decode('ascii')}"


def _boxes_to_masks(boxes: List[List[float]], size: Tuple[int, int]) -> List[Image.Image]:
    # Simple rectangular masks as fallback when SAM is not available
    W, H = size
    masks: List[Image.Image] = []
    for (x0, y0, x1, y1) in boxes:
        m = Image.new("L", (W, H), 0)
        # clamp
        xa = int(max(0, min(W - 1, x0)))
        ya = int(max(0, min(H - 1, y0)))
        xb = int(max(0, min(W, x1)))
        yb = int(max(0, min(H, y1)))
        if xb > xa and yb > ya:
            for y in range(ya, yb):
                for x in range(xa, xb):
                    m.putpixel((x, y), 255)
        masks.append(m)
    return masks


def detect_items(image_url: str, max_items: int = 8) -> List[Dict]:
    """Return a list of { label, box:[x0,y0,x1,y1], mask_url } using OWL-ViT and optional SAM.
    GroundingDINO+SAM2 can be substituted here later without changing the API.
    """
    r = requests.get(image_url, timeout=15)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    W, H = img.size

    owl = _load_owl()
    boxes: List[List[float]] = []
    labels: List[str] = []
    if owl is not None:
        proc, model = owl
        texts = [[p.strip()] for p in CLOTHING_PROMPTS if p.strip()]
        inputs = proc(text=texts, images=img, return_tensors="pt")
        outputs = model(**inputs)
        # convert to boxes on original image
        target_sizes = torch.tensor([[H, W]])  # type: ignore
        logits = outputs.logits[0]
        boxes_pred = outputs.pred_boxes[0]
        # choose top-1 per prompt
        import torch  # type: ignore
        for i in range(logits.shape[0]):
            scores = logits[i].sigmoid()
            v, ix = scores.max(dim=-1)
            if float(v) < 0.2:
                continue
            b = boxes_pred[i, ix]
            b = proc.post_process_object_detection({"scores": v.unsqueeze(0), "boxes": b.unsqueeze(0)}, target_sizes=target_sizes)[0]["boxes"][0]  # type: ignore
            x0, y0, x1, y1 = [float(b[j].item()) for j in range(4)]
            boxes.append([x0, y0, x1, y1])
            labels.append(CLOTHING_PROMPTS[i])
            if len(boxes) >= max_items:
                break
    else:
        # No OWL-ViT; return a central box as placeholder
        side = min(W, H)
        left = (W - side) // 2
        top = (H - side) // 2
        boxes = [[left, top, left + side, top + side]]
        labels = ["garment"]

    # Masks via SAM (if available), else rectangular masks
    sams = _load_sam()
    masks_urls: List[str] = []
    if sams is not None:
        # SAM expects numpy image
        np_img = np.array(img)
        sams.set_image(np_img)
        for (x0, y0, x1, y1) in boxes:
            box_np = np.array([x0, y0, x1, y1])
            try:
                m, _, _ = sams.predict(box=box_np, multimask_output=False)
                mask = (m[0] * 255).astype(np.uint8)
                pil = Image.fromarray(mask, mode="L")
            except Exception:
                pil = _boxes_to_masks([[x0, y0, x1, y1]], (W, H))[0]
            masks_urls.append(_to_data_url(pil))
    else:
        for m in _boxes_to_masks(boxes, (W, H)):
            masks_urls.append(_to_data_url(m))

    out: List[Dict] = []
    for i, b in enumerate(boxes):
        out.append({
            "label": labels[i] if i < len(labels) else f"item_{i+1}",
            "box": b,
            "maskUrl": masks_urls[i] if i < len(masks_urls) else None,
        })
    return out


def extract_best_garment(image_url: str, category_hint: Optional[str] = None) -> Dict:
    items = detect_items(image_url, max_items=8)
    # score by area and category match if hint provided
    best = None
    best_score = -1.0
    for it in items:
        (x0, y0, x1, y1) = it.get("box", [0, 0, 0, 0])
        area = max(1.0, (x1 - x0) * (y1 - y0))
        score = area
        if category_hint and category_hint.lower() in str(it.get("label", "")).lower():
            score *= 1.5
        if score > best_score:
            best = it
            best_score = score
    if not best:
        return {"ok": False, "error": "no items found"}
    return {"ok": True, "maskUrl": best.get("maskUrl"), "label": best.get("label")}

