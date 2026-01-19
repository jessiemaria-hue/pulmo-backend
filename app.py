import os
import io
import base64
from typing import Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image
from scipy.io import loadmat

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

# Prevent Ultralytics permission issue
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
# MODEL LOADING
# ======================================================

def load_model():
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
        print("[PulmoLens] Model loaded:", model_path)

    except Exception as e:
        model = None
        model_error = str(e)
        print("[PulmoLens] Model load failed:", model_error)


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

        # windowing (match training)
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


def masks_from_ultralytics(result) -> np.ndarray:
    if result.masks is None or result.masks.data is None:
        return np.zeros(result.orig_shape, dtype=np.uint8)
    m = result.masks.data.cpu().numpy()
    return (m.sum(axis=0) > 0).astype(np.uint8)


def confidence_from_result(result) -> Optional[float]:
    try:
        if result.boxes is None or result.boxes.conf is None:
            return None
        confs = result.boxes.conf.detach().cpu().numpy()
        if confs.size == 0:
            return None
        return float(confs.max())
    except Exception:
        return None


# ---------- GT READERS ----------

def read_gt_txt_yolo_seg_as_binary(gt_txt_bytes: bytes, target_hw: tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    mask = np.zeros((H, W), dtype=np.uint8)

    text = gt_txt_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return mask

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        coords = parts[1:]
        if len(coords) % 2 != 0:
            continue

        pts = []
        for i in range(0, len(coords), 2):
            x = float(coords[i])
            y = float(coords[i + 1])
            px = int(round(x * (W - 1)))
            py = int(round(y * (H - 1)))
            pts.append([px, py])

        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)

    return mask


def read_gt_mat_slice_as_binary(
    gt_mat_bytes: bytes,
    target_hw: tuple[int, int],
    slice_idx: int,
) -> np.ndarray:
    H, W = target_hw
    m = loadmat(io.BytesIO(gt_mat_bytes))

    if "Mask" in m:
        vol = m["Mask"]
    elif "mask" in m:
        vol = m["mask"]
    else:
        raise HTTPException(400, "MAT read failed: mask key not found")

    if vol.ndim != 3:
        raise HTTPException(400, f"Expected 3D mask, got {vol.ndim}D")

    if not (0 <= slice_idx < vol.shape[2]):
        raise HTTPException(400, "slice_idx out of range")

    gt2d = (vol[:, :, slice_idx] > 0).astype(np.uint8)

    if gt2d.shape != (H, W):
        gt2d = cv2.resize(gt2d, (W, H), interpolation=cv2.INTER_NEAREST)

    return gt2d


def mask2d_to_yolo_txt(mask: np.ndarray) -> str:
    H, W = mask.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    lines = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        coords = []
        for p in cnt.squeeze():
            x = p[0] / (W - 1)
            y = p[1] / (H - 1)
            coords.extend([x, y])
        if coords:
            lines.append("0 " + " ".join(f"{c:.6f}" for c in coords))
    return "\n".join(lines)

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
    file: UploadFile = File(...),               # .dcm or .png
    gt_mat: Optional[UploadFile] = File(None),  # MODE B
    gt_txt: Optional[UploadFile] = File(None),  # MODE A fallback
    slice_idx: Optional[int] = Query(None),     # REQUIRED for MODE B
    filename: str = Form("input"),
    include_images: bool = Query(False),
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

    # Image ingest
    if name.endswith(".dcm"):
        img_pil = read_dicom_as_rgb(raw)
    else:
        img_pil = read_png_like(raw)

    img_np = np.array(img_pil)

    # Inference
    try:
        res = model(img_np, conf=CONF_THRES, verbose=False)[0]
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": f"Inference failed: {e}"},
            status_code=500,
        )

    pred_mask = masks_from_ultralytics(res)
    conf = confidence_from_result(res)

    payload: Dict[str, Any] = {
        "filename": file.filename or filename,
        "has_gt": bool(gt_mat or gt_txt),
        "confidence": conf,
        "iou": None,
    }

    gt_bin = None

    # MODE B: .dcm + .mat (+ slice_idx)
    if gt_mat:
        if slice_idx is None:
            raise HTTPException(400, "slice_idx is required for gt_mat")

        gt2d = read_gt_mat_slice_as_binary(
            await gt_mat.read(),
            pred_mask.shape,
            slice_idx,
        )

        yolo_txt = mask2d_to_yolo_txt(gt2d)

        gt_bin = read_gt_txt_yolo_seg_as_binary(
            yolo_txt.encode(),
            pred_mask.shape
        )

    # MODE A: .txt directly
    elif gt_txt:
        gt_bin = read_gt_txt_yolo_seg_as_binary(
            await gt_txt.read(),
            pred_mask.shape
        )

    if gt_bin is not None:
        payload["iou"] = compute_iou(pred_mask, gt_bin)

    if include_images:
        pred_overlay = overlay_mask(img_np, pred_mask)
        payload["input_png"] = pil_to_b64_png(Image.fromarray(img_np))
        payload["pred_overlay_png"] = pil_to_b64_png(Image.fromarray(pred_overlay))

    return payload
