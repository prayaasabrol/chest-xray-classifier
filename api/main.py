from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.predict import ChestXrayPredictor

DISCLAIMER = "Educational project only. Not a medical device and not for diagnosis."
COLOR_THRESHOLD = 8  # tune this if real X-rays get rejected or photos slip through


def looks_like_xray(image: Image.Image) -> bool:
    """Rough check: X-rays are grayscale, so R, G and B should be nearly identical."""
    rgb = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.int16)
    diff = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    return diff < COLOR_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = ChestXrayPredictor("checkpoints/best_model.pt")
    yield


app = FastAPI(title="Chest X-ray Classifier", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Chest X-ray Classifier API. See /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(app.state.predictor.device)}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload a valid image file (PNG or JPG).")

    if not looks_like_xray(image):
        raise HTTPException(
            status_code=422,
            detail="This doesn't look like a chest X-ray (the image contains colour). Upload a grayscale chest X-ray.",
        )

    scores = app.state.predictor.predict(image)
    return {"predictions": scores, "disclaimer": DISCLAIMER}
