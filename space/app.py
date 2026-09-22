"""
Maize Leaf Diagnostic: a small prediction service around the fine-tuned
MobileNetV2 from experiment E9.

The preprocessing here mirrors the notebook exactly: centre square crop,
resize to 128x128 with bilinear interpolation, scale into [0, 1]. Each photo
is read twice, once as sent and once mirrored, and the two answers are
averaged.

Endpoints
    GET  /          the web app
    GET  /health    model state and metadata
    POST /predict   multipart form field "file" -> class probabilities
"""

import io
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

MODEL_PATH = os.environ.get("MODEL_PATH", "maize_e9.keras")
IMG_SIZE = (128, 128)
MAX_BYTES = 12 * 1024 * 1024

# LabelEncoder order from the notebook. Do not reorder.
CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
LABELS = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

# Confidence bands used by the web app.
SURE, MAYBE = 0.80, 0.55

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Maize Leaf Diagnostic", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = None
model_error = None
started = time.time()


@app.on_event("startup")
def load_model() -> None:
    """Load the Keras model once, at boot, so the first request is not slow."""
    global model, model_error
    if not os.path.exists(MODEL_PATH):
        model_error = f"Model file {MODEL_PATH} was not found in the Space."
        return
    try:
        import keras

        try:
            model = keras.saving.load_model(MODEL_PATH, compile=False)
        except Exception:
            # The E9 graph carries preprocess_input as a lambda, which Keras
            # refuses to deserialise under safe mode.
            model = keras.saving.load_model(MODEL_PATH, compile=False, safe_mode=False)
        model.predict(np.zeros((1, *IMG_SIZE, 3), dtype="float32"), verbose=0)
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        model_error = f"The model could not be loaded: {exc}"


def prepare(raw: bytes) -> np.ndarray:
    """Bytes of a photo -> a batch of two 128x128 images, the second mirrored."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB")

    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # Downscale in two steps so a large phone photo does not alias, the way
    # the 256px dataset images did not when the notebook resized them.
    if side > 512:
        img = img.resize((256, 256), Image.LANCZOS)
    img = img.resize(IMG_SIZE, Image.BILINEAR)

    x = np.asarray(img, dtype="float32") / 255.0
    return np.stack([x, x[:, ::-1, :]])


def green_fraction(batch: np.ndarray) -> float:
    """Share of leaf-green pixels, used to spot photos that hold no leaf."""
    rgb = batch[0]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    chroma = mx - mn
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(mx > 0, chroma / np.maximum(mx, 1e-6), 0.0)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        hue = np.select(
            [mx == r, mx == g, mx == b],
            [
                ((g - b) / np.maximum(chroma, 1e-6)) % 6,
                (b - r) / np.maximum(chroma, 1e-6) + 2,
                (r - g) / np.maximum(chroma, 1e-6) + 4,
            ],
        ) * 60.0
    hue = np.where(hue < 0, hue + 360, hue)
    leafy = (mx > 0.15) & (sat > 0.15) & (hue >= 55) & (hue <= 170)
    return float(leafy.mean())


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ready": model is not None,
            "error": model_error,
            "model": "MobileNetV2 (fine-tuned, E9)",
            "input": {"size": list(IMG_SIZE), "scaling": "[0, 1]"},
            "classes": LABELS,
            "test_accuracy": 0.9316,
            "test_macro_f1": 0.9117,
            "uptime_seconds": round(time.time() - started),
            "version": app.version,
        }
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="That photo is too large. Keep it under 12 MB.")

    try:
        batch = prepare(raw)
    except Exception:  # noqa: BLE001 - any decode failure means a bad photo
        raise HTTPException(status_code=400, detail="That file could not be read as a photo.")

    t0 = time.perf_counter()
    probs = model.predict(batch, verbose=0).mean(axis=0)
    elapsed = round((time.perf_counter() - t0) * 1000)

    top = int(np.argmax(probs))
    confidence = float(probs[top])
    band = "sure" if confidence >= SURE else "maybe" if confidence >= MAYBE else "unsure"

    return JSONResponse(
        {
            "label": LABELS[top],
            "class_id": CLASSES[top],
            "confidence": confidence,
            "band": band,
            "probabilities": {LABELS[i]: float(p) for i, p in enumerate(probs)},
            "green_fraction": green_fraction(batch),
            "inference_ms": elapsed,
            "note": "A helper, not a final diagnosis. Confirm with an agricultural extension officer.",
        }
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/class_samples.png")
def samples() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "class_samples.png"))
