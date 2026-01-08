import os
import io
import base64
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ----------------------------
# Config
# ----------------------------
FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "https://pulmo-lens.netlify.app"
).split(",")

CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))

# Where we store model inside container
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/models")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "best.pt"))

# Hugging Face repo for model
HF_REPO_ID = os.getenv("HF_REPO_ID", "YOUR_USERNAME/YOUR_MODEL_REPO")  # contoh: jessiemaria/pulmo-lens-model
HF_FILENAME = os.getenv("HF_FILENAME", "best.pt")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()  # set in Railway Variables (secret)

# prevent Ultralytics warning
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

app = FastAPI(title="PulmoLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONTEND_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_error = None

# ----------------------------
# Model download + load
# ----------------------------
def ensure_model_file():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=HF_TOKEN if HF_TOKEN else None,
        cache_dir=MODEL_DIR,
    )

    # copy to fixed path
    with open(local_path, "rb") as src, open(MODEL_PATH, "wb") as dst:
        dst.write(src.read())

    return MODEL_PATH

def load_model():
    global model, model_error
    if model is not None:
        return
    try:
        path = ensure_model_file()
        model = YOLO(path)
        model_error = None
        print("[PulmoLens] Model loaded OK")
    except Exception as e:
        model = None
        model_error = str(e)
        print(f"[PulmoLens] Model load failed: {model_error}")

@app.on_event("startup")
def startup_event():
    # Try to load model at boot, but DON'T crash the server if it fails
    load_model()

# ----------------------------
# Utils
# ----------------------------
def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def read_png_like(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image (PNG/JPG/JPEG).")

def read_dicom_as_rgb(file_bytes: bytes) -> Image.Image:
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)

        def _to_float(x):
            try:
                return float(x[0]) if hasattr(x, "__len__") and not isinstance(x, (str, bytes)) else float(x)
            except Exception:
                return float(x)

        if wc is not None and ww is not None:
            wc = _to_float(wc)
            ww = _to_float(ww)
            lo = wc - ww / 2.0
            hi = wc + ww / 2.0
            arr = np.clip(arr, lo, hi)
        else:
            p1, p99 = np.percentile(arr, (1, 99))
            arr = np.clip(arr, p1, p99)

        arr = arr - arr.min()
        denom = (arr.max() - arr.min()) + 1e-8
        arr = (arr / denom * 255.0).clip(0, 255).astype(np.uint8)

        rgb = np.stack([arr, arr, arr], axis=-1)
        return Image.fromarray(rgb, mode="RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read DICOM: {e}")

def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 180, 0), alpha=0.35) -> np.ndarray:
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
    if result.masks is None or result.masks.data is None:
        return None
    m = result.masks.data.cpu().numpy()
    return (m.sum(axis=0) > 0).astype(np.uint8)

def boxes_conf(result) -> float:
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0
    confs = result.boxes.conf
    if confs is None:
        return 0.0
    confs = confs.cpu().numpy()
    return float(np.mean(confs)) if len(confs) else 0.0

def det_count(result) -> int:
    return int(len(result.boxes)) if result.boxes is not None else 0

# ----------------------------
# Routes
# ----------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_error,
        "hf_repo_id": HF_REPO_ID,
        "hf_filename": HF_FILENAME,
        "has_hf_token": bool(HF_TOKEN),
        "conf_thres": CONF_THRES,
    }

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    gt_mask: Optional[UploadFile] = File(None),
    filename: str = Form("input"),
):
    global model
    if model is None:
        load_model()
    if model is None:
        return JSONResponse({"error": f"Model not loaded: {model_error}"}, status_code=503)

    raw = await file.read()
    name = (file.filename or filename or "input").lower()

    if name.endswith(".dcm"):
        img_pil = read_dicom_as_rgb(raw)
    else:
        img_pil = read_png_like(raw)

    img_np = np.array(img_pil)

    try:
        res = model(img_np, verbose=False, conf=CONF_THRES)[0]
    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {e}"}, status_code=500)

    pred_mask = masks_from_ultralytics(res)
    if pred_mask is None:
        pred_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

    pred_overlay = overlay_mask(img_np, pred_mask, color=(255, 180, 0), alpha=0.35)

    payload = {
        "filename": file.filename or filename or "input",
        "confidence": boxes_conf(res),
        "detections": det_count(res),
        "input_png": pil_to_b64_png(Image.fromarray(img_np)),
        "pred_overlay_png": pil_to_b64_png(Image.fromarray(pred_overlay)),
    }

    if gt_mask is not None:
        gt_raw = await gt_mask.read()
        try:
            gt_img = Image.open(io.BytesIO(gt_raw)).convert("L")
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot decode GT mask image.")

        gt_arr = (np.array(gt_img) > 127).astype(np.uint8)
        if gt_arr.shape != pred_mask.shape:
            gt_img = gt_img.resize((pred_mask.shape[1], pred_mask.shape[0]))
            gt_arr = (np.array(gt_img) > 127).astype(np.uint8)

        payload["gt_overlay_png"] = pil_to_b64_png(Image.fromarray(overlay_mask(img_np, gt_arr, color=(0, 180, 255), alpha=0.35)))
        payload["iou"] = compute_iou(pred_mask, gt_arr)

    return payload
