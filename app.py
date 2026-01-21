import os, io, base64, uuid, logging
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
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "512"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/models")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")
HF_FILENAME = os.getenv("HF_FILENAME", "best.pt")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# ======================================================
# APP
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
# MODEL
# ======================================================

def load_model():
    global model, model_error
    if model is not None:
        return
    try:
        if not HF_REPO_ID:
            raise RuntimeError("HF_REPO_ID missing")

        os.makedirs(MODEL_DIR, exist_ok=True)
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=HF_TOKEN if HF_TOKEN else None,
            cache_dir=MODEL_DIR,
        )

        model = YOLO(path)
        model_error = None
        log.info("Model loaded: %s", path)

    except Exception as e:
        model = None
        model_error = str(e)
        log.exception("Model load failed")

@app.on_event("startup")
def startup():
    load_model()

# ======================================================
# UTILITIES
# ======================================================

def guard(b: bytes, label: str):
    if len(b) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{label} too large")

def read_png(b: bytes):
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image")

def read_dcm(b: bytes):
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(b), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept
    p1, p99 = np.percentile(arr, (1, 99))
    arr = np.clip(arr, p1, p99)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr]*3, axis=-1))

def mask_from_yolo(txt: bytes, hw: Tuple[int,int]):
    H,W = hw
    m = np.zeros((H,W),np.uint8)
    for ln in txt.decode().splitlines():
        p = ln.split()[1:]
        pts=[]
        for i in range(0,len(p),2):
            pts.append([int(float(p[i])*(W-1)), int(float(p[i+1])*(H-1))])
        if len(pts)>=3:
            cv2.fillPoly(m,[np.array(pts,np.int32)],1)
    return m

def pick_volume(mat):
    for k in mat:
        if not k.startswith("__"):
            v=mat[k]
            if isinstance(v,np.ndarray) and v.ndim==3:
                return v
    raise HTTPException(400,"MAT has no 3D mask")

def mat_slice(b: bytes, idx:int, hw):
    vol = pick_volume(loadmat(io.BytesIO(b)))
    Ht,Wt=hw
    dims=list(vol.shape)
    pairs=[]
    for a in range(3):
        for b in range(a+1,3):
            score=abs(dims[a]-Ht)+abs(dims[b]-Wt)
            pairs.append((score,a,b))
    _,ah,aw=min(pairs)
    as_ = ({0,1,2}-{ah,aw}).pop()
    vol=np.moveaxis(vol,(ah,aw,as_),(0,1,2))
    if not 0<=idx<vol.shape[2]:
        raise HTTPException(400,"slice_idx out of range")
    m=(vol[:,:,idx]>0).astype(np.uint8)
    return cv2.resize(m,(Wt,Ht),interpolation=cv2.INTER_NEAREST)

def iou(a,b):
    return float(np.logical_and(a,b).sum()/np.logical_or(a,b).sum()) if np.logical_or(a,b).sum()>0 else 0.0

# ======================================================
# ROUTES
# ======================================================

@app.get("/api/health")
def health():
    return {"status":"ok","model_loaded":model is not None,"model_error":model_error,"imgsz":YOLO_IMGSZ}

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    gt_txt: Optional[UploadFile] = File(None),
    gt_mat: Optional[UploadFile] = File(None),
    slice_idx: Optional[int] = Query(None),
):
    if model is None:
        load_model()
    if model is None:
        raise HTTPException(503, model_error)

    raw=await file.read(); guard(raw,"file")
    img = read_dcm(raw) if file.filename.lower().endswith(".dcm") else read_png(raw)
    img=np.array(img)
    img=cv2.resize(img,(YOLO_IMGSZ,YOLO_IMGSZ))

    res=model(img,conf=CONF_THRES,imgsz=YOLO_IMGSZ,verbose=False)[0]
    pred=(res.masks.data.cpu().numpy().sum(0)>0).astype(np.uint8)

    gt=None
    if gt_txt:
        gt=mask_from_yolo(await gt_txt.read(),(YOLO_IMGSZ,YOLO_IMGSZ))
    elif gt_mat:
        if slice_idx is None:
            raise HTTPException(400,"slice_idx required")
        gt2=mat_slice(await gt_mat.read(),slice_idx,(YOLO_IMGSZ,YOLO_IMGSZ))
        gt=gt2

    return {
        "confidence": float(res.boxes.conf.max()) if res.boxes else None,
        "iou": iou(pred,gt) if gt is not None else None,
        "gt_source": "txt" if gt_txt else "mat",
    }
