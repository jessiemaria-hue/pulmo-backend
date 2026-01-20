import os
import io
import base64
import uuid
import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image
from scipy.io import loadmat

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ======================================================
# CONFIG
# ======================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pulmolens")

FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "https://pulmo-lens.netlify.app"
).split(",")

CONF_THRES = float(os.getenv("CONF_THRES", "0.25"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "512"))  # set 512 to match training
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10MB default

MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/models")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")
HF_FILENAME = os.getenv("HF_FILENAME", "best.pt")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Prevent Ultralytics permission issue (Railway)
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# ======================================================
# APP INIT
# ======================================================

app = FastAPI(title="PulmoLens API", version="1.0.0")

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
        log.info("[PulmoLens] Model loaded: %s", model_path)

    except Exception as e:
        model = None
        model_error = str(e)
        log.exception("[PulmoLens] Model load failed: %s", model_error)


@app.on_event("startup")
def startup_event():
    load_model()

# ======================================================
# UTILITIES
# ======================================================

def _guard_size(b: bytes, label: str):
    if len(b) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label} too large: {len(b)} bytes (max {MAX_UPLOAD_BYTES})"
        )


def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_png_like(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file (expected PNG/JPG)")


def read_dicom_as_rgb(file_bytes: bytes) -> Image.Image:
    """
    Minimal DICOM -> 8-bit RGB conversion. Matches your training-style normalization (percentile clipping).
    """
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


def masks_from_ultralytics(result, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Returns a single binary mask. If no mask, returns zeros of target_hw.
    """
    H, W = target_hw
    if result.masks is None or result.masks.data is None:
        return np.zeros((H, W), dtype=np.uint8)

    m = result.masks.data.detach().cpu().numpy()  # [N, H, W]
    if m.ndim != 3:
        return np.zeros((H, W), dtype=np.uint8)

    merged = (m.sum(axis=0) > 0).astype(np.uint8)
    if merged.shape != (H, W):
        merged = cv2.resize(merged, (W, H), interpolation=cv2.INTER_NEAREST)
    return merged


def confidence_from_result(result) -> Optional[float]:
    try:
        if result.boxes is None or result.boxes.conf is None:
            return None
        confs = result.boxes.conf.detach().cpu().numpy()
        return float(confs.max()) if confs.size else None
    except Exception:
        return None


# ======================================================
# GT (YOLO POLYGON TXT) -> BINARY MASK @ 512x512
# ======================================================

def read_gt_txt_yolo_seg_as_binary(gt_txt_bytes: bytes, target_hw: Tuple[int, int]) -> np.ndarray:
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


# ======================================================
# MAT (3D) -> SLICE 2D @ target size, then optional polygon convert
# ======================================================

def _pick_volume_3d(mat: Dict[str, Any]) -> np.ndarray:
    # Prefer common keys first
    for k in ("Mask", "mask", "GT", "gt", "Label", "label"):
        v = mat.get(k)
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v

    # Fallback: first 3D array
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v

    raise HTTPException(status_code=400, detail="MAT read failed: no 3D volume found")


def mat_slice_to_binary(
    gt_mat_bytes: bytes,
    slice_idx: int,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Robust: supports volumes shaped (H,W,S) or (S,H,W) or (H,S,W) etc.
    Heuristic: choose two axes closest to target H/W as spatial axes; remaining axis is slice.
    """
    Ht, Wt = target_hw
    m = loadmat(io.BytesIO(gt_mat_bytes))
    vol = _pick_volume_3d(m)

    # Find which two axes look like spatial dims (~Ht, ~Wt)
    dims = list(vol.shape)  # [d0,d1,d2]
    axis_scores = []
    for a in range(3):
        for b in range(a + 1, 3):
            da, db = dims[a], dims[b]
            score = abs(da - Ht) + abs(db - Wt)
            axis_scores.append((score, a, b))

    axis_scores.sort(key=lambda x: x[0])
    _, ax_h, ax_w = axis_scores[0]
    ax_s = ({0, 1, 2} - {ax_h, ax_w}).pop()

    S = dims[ax_s]
    if not (0 <= slice_idx < S):
        raise HTTPException(status_code=400, detail=f"slice_idx out of range (0..{S-1})")

    # Move axes to (H, W, S)
    vol_hw_s = np.moveaxis(vol, (ax_h, ax_w, ax_s), (0, 1, 2))  # (H?, W?, S)

    gt2d = (vol_hw_s[:, :, slice_idx] > 0).astype(np.uint8)

    # Resize to target exactly
    if gt2d.shape != (Ht, Wt):
        gt2d = cv2.resize(gt2d, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

    return gt2d


def mask2d_to_yolo_txt(mask: np.ndarray) -> str:
    """
    Converts a 2D binary mask to YOLO polygon txt format.
    NOTE: This is inherently lossy vs raw MAT mask; use for compatibility only.
    """
    H, W = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        coords = []
        pts = cnt.squeeze()
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        for p in pts:
            x = float(p[0]) / float(W - 1)
            y = float(p[1]) / float(H - 1)
            coords.extend([x, y])
        if coords:
            lines.append("0 " + " ".join(f"{c:.6f}" for c in coords))
    return "\n".join(lines)

# ======================================================
# ROUTES
# ======================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_error,
        "imgsz": YOLO_IMGSZ,
        "conf_thres": CONF_THRES,
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),               # .dcm or .png/.jpg
    gt_txt: Optional[UploadFile] = File(None),  # preferred GT for eval (production)
    gt_mat: Optional[UploadFile] = File(None),  # optional GT source; requires slice_idx
    slice_idx: Optional[int] = Query(None),     # required if gt_mat provided
    include_images: bool = Query(False),
    filename: str = Form("input"),
):
    """
    Production behavior:
    - Inference always runs at YOLO_IMGSZ x YOLO_IMGSZ (default 512).
    - IoU is computed in the same space (512) for consistency with training.
    - Overlay is returned in original resolution (optional).
    """
    global model

    req_id = str(uuid.uuid4())[:8]

    if model is None:
        load_model()
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": f"Model not available: {model_error}", "request_id": req_id},
        )

    raw = await file.read()
    _guard_size(raw, "file")
    name = (file.filename or filename or "input").lower()

    # --- ingest image ---
    img_pil = read_dicom_as_rgb(raw) if name.endswith(".dcm") else read_png_like(raw)
    img_np = np.array(img_pil)
    orig_h, orig_w = img_np.shape[:2]

    # --- resize to imgsz (match YOLO training/eval space) ---
    img_512 = cv2.resize(img_np, (YOLO_IMGSZ, YOLO_IMGSZ), interpolation=cv2.INTER_LINEAR)

    # --- inference ---
    try:
        # imgsz param keeps ultralytics consistent if it applies internal transforms
        res = model(img_512, conf=CONF_THRES, imgsz=YOLO_IMGSZ, verbose=False)[0]
    except Exception as e:
        log.exception("[%s] Inference failed", req_id)
        return JSONResponse(status_code=500, content={"status": "error", "error": f"Inference failed: {e}", "request_id": req_id})

    pred_mask_512 = masks_from_ultralytics(res, (YOLO_IMGSZ, YOLO_IMGSZ))
    conf = confidence_from_result(res)

    payload: Dict[str, Any] = {
        "request_id": req_id,
        "filename": file.filename or filename,
        "has_gt": bool(gt_txt or gt_mat),
        "confidence": conf,
        "iou": None,
        "imgsz": YOLO_IMGSZ,
    }

    # --- GT handling (IoU computed in 512-space) ---
    gt_bin_512 = None

    if gt_txt is not None:
        gt_bytes = await gt_txt.read()
        _guard_size(gt_bytes, "gt_txt")
        gt_bin_512 = read_gt_txt_yolo_seg_as_binary(gt_bytes, (YOLO_IMGSZ, YOLO_IMGSZ))

    elif gt_mat is not None:
        if slice_idx is None:
            raise HTTPException(status_code=400, detail="slice_idx is required when gt_mat is provided")

        mat_bytes = await gt_mat.read()
        _guard_size(mat_bytes, "gt_mat")

        # Read MAT slice -> 2D mask @ 512
        gt2d_512 = mat_slice_to_binary(mat_bytes, slice_idx=slice_idx, target_hw=(YOLO_IMGSZ, YOLO_IMGSZ))

        # Convert to YOLO polygon then back to mask (compat mode)
        # NOTE: This is lossy vs direct pixel IoU. Use gt_txt for authoritative evaluation.
        yolo_txt = mask2d_to_yolo_txt(gt2d_512)
        gt_bin_512 = read_gt_txt_yolo_seg_as_binary(yolo_txt.encode("utf-8"), (YOLO_IMGSZ, YOLO_IMGSZ))

        payload["slice_idx"] = slice_idx
        payload["gt_source"] = "mat->poly->mask (lossy)"
    else:
        payload["gt_source"] = None

    if gt_bin_512 is not None:
        payload["iou"] = compute_iou(pred_mask_512, gt_bin_512)

    # --- overlay (optional, in original resolution for UI) ---
    if include_images:
        pred_mask_orig = cv2.resize(pred_mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        overlay = overlay_mask(img_np, pred_mask_orig)
        payload["input_png"] = pil_to_b64_png(Image.fromarray(img_np))
        payload["pred_overlay_png"] = pil_to_b64_png(Image.fromarray(overlay))

    return payload
