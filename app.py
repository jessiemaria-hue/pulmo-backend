import os
import io
import base64
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ======================================================
# CONFIG
# ======================================================

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "https://pulmo-lens.netlify.app"
).split(",")

CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))

MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/models")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")
HF_FILENAME = os.getenv("HF_FILENAME", "best.pt")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Prevent Ultralytics config permission issue on Railway/containers
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# ======================================================
# APP INIT
# ======================================================

app = FastAPI(title="PulmoLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONTEND_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[YOLO] = None
model_error: Optional[str] = None


# ======================================================
# MODEL LOADING (SAFE)
# ======================================================

def load_model():
    """Download + load model. Must not crash server if it fails."""
    global model, model_error

    if model is not None:
        return

    try:
        if not HF_REPO_ID:
            raise RuntimeError("HF_REPO_ID is not set")

        os.makedirs(MODEL_DIR, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN if HF_TOKEN else None,
            cache_dir=MODEL_DIR,
        )

        model = YOLO(model_path)
        model_error = None
        print("[PulmoLens] Model loaded successfully:", model_path)

    except Exception as e:
        model = None
        model_error = str(e)
        print(f"[PulmoLens] Model load failed: {model_error}")


@app.on_event("startup")
def startup_event():
    load_model()


# ======================================================
# UTILITIES
# ======================================================

def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_png_like(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


def read_dicom_as_rgb(file_bytes: bytes) -> Image.Image:
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        p1, p99 = np.percentile(arr, (1, 99))
        arr = np.clip(arr, p1, p99)

        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)

        rgb = np.stack([arr, arr, arr], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DICOM read failed: {e}")


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 180, 0), alpha=0.35):
    out = rgb.copy()
    m = mask.astype(bool)
    if m.any():
        overlay = out.copy()
        overlay[m] = np.array(color, dtype=np.uint8)
        out = (out * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return out


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def masks_from_ultralytics(result) -> Optional[np.ndarray]:
    # returns HxW uint8 mask (0/1) or None
    if result.masks is None or result.masks.data is None:
        return None
    m = result.masks.data.cpu().numpy()  # (N, H, W)
    return (m.sum(axis=0) > 0).astype(np.uint8)


def confidence_from_result(result) -> Optional[float]:
    # quick summary confidence: max box confidence if exists
    try:
        if result.boxes is None or result.boxes.conf is None:
            return None
        confs = result.boxes.conf.detach().cpu().numpy()
        if confs.size == 0:
            return None
        return float(confs.max())
    except Exception:
        return None


def read_gt_mask_as_binary(gt_bytes: bytes, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Reads GT mask image, converts to binary, and resizes to match pred mask.
    target_hw: (H, W)
    """
    gt_img = Image.open(io.BytesIO(gt_bytes)).convert("L")

    # Resize with NEAREST to avoid destroying labels
    if gt_img.size != (target_hw[1], target_hw[0]):
        gt_img = gt_img.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)

    gt_arr = np.array(gt_img)

    # Robust binarization:
    # If mask is inverted (lesion=0, background=255), user should fix input,
    # but we can still threshold normally here.
    gt_bin = (gt_arr > 127).astype(np.uint8)
    return gt_bin


# ======================================================
# ROUTES
# ======================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_error,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    gt_mask: Optional[UploadFile] = File(None),
    filename: str = Form("input"),
    include_images: bool = Query(False, description="Return base64 PNGs if true"),
):
    global model

    if model is None:
        load_model()

    if model is None:
        return JSONResponse(
            {"status": "error", "error": f"Model not available: {model_error}"},
            status_code=503,
        )

    raw = await file.read()
    name = (file.filename or filename or "input").lower()

    if name.endswith(".dcm"):
        img_pil = read_dicom_as_rgb(raw)
    else:
        img_pil = read_png_like(raw)

    img_np = np.array(img_pil)  # HxWx3

    try:
        res = model(img_np, conf=CONF_THRES, verbose=False)[0]
    except Exception as e:
        return JSONResponse({"status": "error", "error": f"Inference failed: {e}"}, status_code=500)

    pred_mask = masks_from_ultralytics(res)
    if pred_mask is None:
        pred_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

    conf = confidence_from_result(res)

    payload: Dict[str, Any] = {
        "filename": file.filename or filename,
        "has_gt": gt_mask is not None,
        "confidence": conf,
        "iou": None,
    }

    if gt_mask is not None:
        gt_raw = await gt_mask.read()
        gt_bin = read_gt_mask_as_binary(gt_raw, target_hw=pred_mask.shape)
        payload["iou"] = compute_iou(pred_mask, gt_bin)

    if include_images:
        pred_overlay = overlay_mask(img_np, pred_mask)
        payload["input_png"] = pil_to_b64_png(Image.fromarray(img_np))
        payload["pred_overlay_png"] = pil_to_b64_png(Image.fromarray(pred_overlay))

    return payload
